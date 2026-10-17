//! # Immix: A mark-region garbage collector
//!
//! The Immix collector is a collector based on mark-and-sweep with a
//! modified thread-local allocation algorithm and a heap layout more optimised for
//! modern CPU caches. The heap is organised into large blocks containing lines
//! which then contain actual objects. The line size is chosen to more or less align
//! with the cache line sizes of modern CPU architectures, such as x86 64. Objects
//! are not collected unless all objects in a line are unreachable, and surprisingly this
//! coarser granularity leads to performance improvements.
//! You can find more information about Immix in this [paper](https://users.cecs.anu.edu.au/~steveb/pubs/papers/immix-pldi-2008.pdf)

use crate::{
    api::{vtable_of, Collectable, Gc, HeapObjectHeader, Trace, Visitor, Weak, GC_BLACK, GC_WHITE},
    bitmap::SpaceBitmap,
    gc_base::{
        AllocationSpace, GcBase, MarkingConstraint, MarkingConstraintRuns, NoHelp,
        NoOpStackDecoder, NoReadBarrier, StackValueDecoder,
    },
    large_space::{LargeObjectSpace, PreciseAllocation},
    make_small_type_id,
    mutator::{approximate_stack_pointer, oom_abort, JoinData, Mutator, MutatorRef, ThreadState},
    safepoint::{GlobalSafepoint, SafepointScope},
    small_type_id,
    utils::{align_usize, formatted_size, stack_bounds::StackBounds},
};
use crate::{
    bitmap::{round_up, ChunkMap},
    gc_base::TLAB,
    utils::align_down,
};
use atomic::Ordering;
use rosalloc::defs::PAGE_SIZE;
use std::{
    any::TypeId, cell::UnsafeCell, marker::PhantomData, mem::size_of, ptr::NonNull, sync::Arc,
};
use std::{
    ptr::null_mut,
    sync::atomic::{AtomicPtr, AtomicUsize},
};
pub fn line_align(ptr: *const u8) -> *mut u8 {
    align_down(ptr as _, IMMIX_LINE_SIZE) as _
}

pub fn is_line_aligned(ptr: *const u8) -> bool {
    line_align(ptr) == ptr as *mut u8
}

pub mod block;
pub mod chunk;
pub mod space;
use block::*;
use chunk::*;
use space::*;

/// Thread local allocator for Immix. This allocator stores two different bump pointers:
/// 1) Large cursor for allocating objects that span multiple lines
/// 2) Regular cursor for objects whose size is smaller than [IMMIX_LINE_SIZE](IMMIX_LINE_SIZE).
pub struct ImmixAllocator {
    cursor: *mut u8,
    limit: *mut u8,
    space: &'static ImmixSpace,
    large_cursor: *mut u8,
    large_limit: *mut u8,
    request_for_large: bool,
    emergency_collection: bool,
    line: Option<*mut u8>,
    bmap: *const SpaceBitmap<8>,
}

impl ImmixAllocator {
    /// Try to acquire recyclable block. Returns false if there is no recyclable blocks or GC threshold is reached.
    pub fn acquire_recyclable_block(&mut self) -> bool {
        if self.is_out_of_memory_on_allocation(IMMIX_BLOCK_SIZE, self.emergency_collection) {
            return false;
        }

        match self.space.get_reusable_block() {
            block if !block.is_null() => {
                self.line = Some(unsafe { (*block).start().add(IMMIX_LINE_SIZE) });
                unsafe {
                    self.space.num_bytes_allocated.fetch_add(
                        match (*block).state() {
                            BlockState::Reusable { unavailable_lines } => {
                                (IMMIX_LINES_PER_BLOCK - unavailable_lines as usize - 1)
                                    * IMMIX_LINE_SIZE
                            }
                            _ => unreachable!(),
                        },
                        Ordering::Relaxed,
                    );
                }
                true
            }
            _ => false,
        }
    }

    /// Acquire recyclable lines from current block. Returns false if there is no more holes in block.
    pub fn acquire_recyclable_lines(&mut self) -> bool {
        while self.line.is_some() || self.acquire_recyclable_block() {
            let line = self.line.unwrap();

            let (start, end) = self.space.acquire_recyclable_lines(line);
            if !start.is_null() && !end.is_null() {
                self.space
                    .num_bytes_allocated
                    .fetch_add(end as usize - start as usize, Ordering::Relaxed);
                self.cursor = start;
                self.limit = end;

                let block = ImmixBlock::align(start).cast::<ImmixBlock>();

                self.line = if unsafe { end == (*block).end() } {
                    // Hole searching reached the end of a reusable block. Set the hole-searching cursor to None.
                    None
                } else {
                    // Update the hole-searching cursor to None.
                    Some(end)
                };
                return true;
            } else {
                // No more recyclable lines. Set the hole-searching cursor to None.
                self.line = None;
            }
        }

        false
    }
    pub fn acquire_clean_block(&mut self) -> bool {
        if self.is_out_of_memory_on_allocation(IMMIX_BLOCK_SIZE, self.emergency_collection) {
            return false;
        }
        match self.space.get_clean_block() {
            block if !block.is_null() => unsafe {
                self.space
                    .num_bytes_allocated
                    .fetch_add(IMMIX_BLOCK_SIZE, Ordering::Relaxed);
                if self.request_for_large {
                    self.large_cursor = (*block).start_address();
                    self.large_limit = (*block).end();
                } else {
                    self.cursor = (*block).start_address();
                    self.limit = (*block).end();
                }

                true
            },
            _ => false,
        }
    }
}

pub trait GetImmixSpace {
    fn immix_space(&self) -> &'static ImmixSpace;
}

impl<H: GcBase<TLAB = Self>> TLAB<H> for ImmixAllocator
where
    H: GetImmixSpace,
{
    fn can_thread_local_allocate(&self, size: usize) -> bool {
        size <= (IMMIX_BLOCK_SIZE >> 1)
    }

    fn refill(&mut self, _mutator: &crate::mutator::MutatorRef<H>, _alloc_size: usize) -> bool {
        if _alloc_size > IMMIX_LINE_SIZE {
            if !self.large_cursor.is_null()
                && unsafe { self.large_cursor.add(_alloc_size) <= self.large_limit }
            {
                return true;
            }

            self.request_for_large = true;
            let acquired = self.acquire_clean_block();
            self.request_for_large = false;
            acquired
        } else {
            if !self.cursor.is_null() && unsafe { self.cursor.add(_alloc_size) <= self.limit } {
                return true;
            }
            self.acquire_recyclable_lines() || self.acquire_clean_block()
        }
    }
    fn allocate<T: crate::api::Collectable + 'static>(
        &mut self,
        value: T,
    ) -> Result<crate::api::Gc<T, H>, T> {
        let size = align_usize(value.allocation_size() + size_of::<HeapObjectHeader>(), 8);
        let (cursor, limit) = if size > IMMIX_LINE_SIZE {
            (&mut self.large_cursor, self.large_limit)
        } else {
            (&mut self.cursor, self.limit)
        };

        if cursor.is_null() || unsafe { cursor.add(size) > limit } {
            return Err(value);
        }

        unsafe {
            let memory = *cursor;
            *cursor = cursor.add(size);
            let header = memory.cast::<HeapObjectHeader>();
            header.write(HeapObjectHeader {
                type_id: small_type_id::<T>(),
                padding: 0,
                padding2: 0,
                value: crate::api::VTable { raw: 0 },
            });
            (*header).set_metadata(vtable_of::<T>());
            (*header).set_size(size);
            ((*header).data() as *mut T).write(value);
            (*self.bmap).set(header.cast());
            Ok(Gc {
                base: NonNull::new_unchecked(header),
                marker: PhantomData,
            })
        }
    }
    fn reset(&mut self) {
        self.large_cursor = null_mut();
        self.large_limit = null_mut();
        self.cursor = null_mut();
        self.limit = null_mut();
        self.line = None;
    }
    fn create(heap: std::sync::Arc<std::cell::UnsafeCell<H>>) -> Self {
        Self {
            space: unsafe { (*heap.get()).immix_space() },
            line: None,
            limit: null_mut(),
            large_cursor: null_mut(),
            large_limit: null_mut(),
            cursor: null_mut(),
            request_for_large: false,
            emergency_collection: false,
            bmap: unsafe { &(*heap.get()).immix_space().mark_bitmap },
        }
    }
}

impl ImmixAllocator {
    #[inline]
    fn is_out_of_memory_on_allocation(&self, alloc_size: usize, grow: bool) -> bool {
        let mut old_target = self.space.target_footprint.load(Ordering::Relaxed);
        loop {
            let old_allocated = self.space.num_bytes_allocated.load(Ordering::Relaxed);
            let new_footprint = old_allocated + alloc_size;
            if new_footprint <= old_target {
                return false;
            } else if new_footprint > self.space.growth_limit {
                return true;
            }

            if grow {
                if let Err(t) = self.space.target_footprint.compare_exchange_weak(
                    old_target,
                    new_footprint,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    old_target = t;
                    //return false;
                } else {
                    return false;
                }
            } else {
                return true;
            }
        }
    }
    pub unsafe fn alloc_slow_once(&mut self, size: usize) -> *mut u8 {
        if self.acquire_clean_block() {
            self.alloc(size)
        } else {
            null_mut()
        }
    }
    #[inline(always)]
    pub unsafe fn alloc_slow_inline(&mut self, size: usize) -> *mut u8 {
        let result = self.alloc_slow_once(size);
        if !result.is_null() {
            return result;
        }

        if self.emergency_collection {
            oom_abort();
        }
        result
    }
    #[inline]
    pub unsafe fn alloc(&mut self, size: usize) -> *mut u8 {
        let result = self.cursor;
        let new_cursor = result.add(size);

        let res = if new_cursor > self.limit {
            if size > IMMIX_LINE_SIZE {
                self.overflow_alloc(size)
            } else {
                self.alloc_slow_hot(size)
            }
        } else {
            self.cursor = new_cursor;
            result
        };
        if !res.is_null() {
            (*self.bmap).set(res);
        }
        res
    }
    #[inline(never)]
    unsafe fn overflow_alloc(&mut self, size: usize) -> *mut u8 {
        let start = self.large_cursor;
        let end = start.add(size);
        if end > self.large_limit {
            self.request_for_large = true;
            let rtn = self.alloc_slow_inline(size);
            self.request_for_large = false;
            rtn
        } else {
            self.large_cursor = end;
            start
        }
    }

    #[cold]
    unsafe fn alloc_slow_hot(&mut self, size: usize) -> *mut u8 {
        if self.acquire_recyclable_lines() {
            self.alloc(size)
        } else {
            self.alloc_slow_inline(size)
        }
    }
}
use parking_lot::{lock_api::RawMutex, RawMutex as Lock};

/// Sticky Immix GC implementation.
///
/// Marking remains full-heap because this collector has no generational write barrier.
/// Selective block evacuation is available as an opt-in fragmentation-reduction step.
pub struct StickyImmix<Decoder: 'static + StackValueDecoder = NoOpStackDecoder> {
    space: &'static ImmixSpace,
    pub(crate) global_heap_lock: Lock,
    pub(crate) large_space_lock: Lock,
    pub(crate) large_space: LargeObjectSpace,
    pub(crate) mutators: Vec<*mut Mutator<Self>>,
    pub(crate) safepoint: GlobalSafepoint,
    pub(crate) mark_stack: Vec<*mut HeapObjectHeader>,
    pub(crate) verbose: u8,
    pub(crate) alloc_color: u8,
    pub(crate) mark_color: u8,
    total_gcs: usize,
    weak_refs: Vec<Weak<dyn Collectable, Self>>,
    constraints: Vec<Box<dyn MarkingConstraint>>,
    growth_multiplier: f64,
    evacuation_enabled: bool,
    evacuation_candidates: Vec<*mut ImmixBlock>,
    evacuation_cursor: *mut u8,
    evacuation_limit: *mut u8,
}

/// Backwards-compatible name for [`StickyImmix`].
pub type Immix<Decoder = NoOpStackDecoder> = StickyImmix<Decoder>;

impl<Decoder: StackValueDecoder> GetImmixSpace for StickyImmix<Decoder> {
    fn immix_space(&self) -> &'static ImmixSpace {
        self.space
    }
}

pub struct StickyImmixOptions {
    /// Determines by how much heap grows after GC cycle. By default set to 1.75
    pub growth_multiplier: f64,
    /// Entire Immix heap area size. Set to 128MB by default, minimal supported value is 4MB (single chunk size)
    pub heap_size: usize,
    /// Initial heap size before triggering GC cycle. By default set to 32MB
    pub initial_size: usize,
    /// Minimal heap size before triggering GC cycle. By default set to 4MB, if `initial_size` is lesser than min_heap_size then it is set to min_heap_size.
    pub min_heap_size: usize,
    /// Maximal heap size before triggering GC cycle. By default set to 128MB
    pub max_heap_size: usize,
    /// Enables verbose loggig to stdout.
    pub verbose: u8,
    /// Evacuate sparse Immix blocks during collection. Disabled by default.
    pub evacuation: bool,
}

/// Backwards-compatible name for [`StickyImmixOptions`].
pub type ImmixOptions = StickyImmixOptions;

impl StickyImmixOptions {
    /// Set growth multiplier. Panics if x <= 1
    pub fn with_growth_multiplier(mut self, x: f64) -> StickyImmixOptions {
        if x <= 1.0 {
            panic!("Growth multiplier is too small")
        }
        self.growth_multiplier = x;
        self
    }

    pub fn with_heap_size(mut self, x: usize) -> StickyImmixOptions {
        if x < 4 * 1024 * 1024 {
            panic!("Heap size too small; Minimal heap size is 1 Immix chunk which is 4MB");
        }
        self.heap_size = x;
        self
    }

    pub fn with_initial_size(mut self, x: usize) -> StickyImmixOptions {
        self.initial_size = x;
        self
    }

    pub fn with_min_heap_size(mut self, x: usize) -> StickyImmixOptions {
        self.min_heap_size = x;
        self
    }

    pub fn with_max_heap_size(mut self, x: usize) -> StickyImmixOptions {
        self.max_heap_size = x;
        self
    }
    pub fn with_verbose(mut self, x: u8) -> StickyImmixOptions {
        self.verbose = x;
        self
    }

    /// Enable or disable opportunistic evacuation of sparse blocks.
    pub fn with_evacuation(mut self, enabled: bool) -> StickyImmixOptions {
        self.evacuation = enabled;
        self
    }
}
impl Default for StickyImmixOptions {
    fn default() -> Self {
        Self {
            growth_multiplier: 1.75,
            heap_size: 128 * 1024 * 1024,
            min_heap_size: 4 * 1024 * 1024,
            max_heap_size: 128 * 1024 * 1024,
            initial_size: 32 * 1024 * 1024,
            verbose: 0,
            evacuation: false,
        }
    }
}

pub fn instantiate_immix<Decoder: StackValueDecoder>(
    options: ImmixOptions,
) -> MutatorRef<Immix<Decoder>> {
    instantiate_sticky_immix(options)
}

pub fn instantiate_sticky_immix<Decoder: StackValueDecoder>(
    options: StickyImmixOptions,
) -> MutatorRef<StickyImmix<Decoder>> {
    let space = Box::leak(Box::new(ImmixSpace::new(
        options.heap_size,
        options.initial_size,
        options.min_heap_size,
        options.max_heap_size,
        options.verbose > 0,
    )));
    space.init_bitmap();
    let immix = Arc::new(UnsafeCell::new(StickyImmix {
        space,
        large_space: LargeObjectSpace::new(),
        large_space_lock: Lock::INIT,
        verbose: options.verbose,
        global_heap_lock: Lock::INIT,

        mutators: vec![],
        safepoint: GlobalSafepoint::new(),
        alloc_color: GC_WHITE,
        mark_color: GC_BLACK,
        mark_stack: Vec::new(),
        total_gcs: 0,
        weak_refs: vec![],
        constraints: vec![],
        growth_multiplier: options.growth_multiplier,
        evacuation_enabled: options.evacuation,
        evacuation_candidates: Vec::new(),
        evacuation_cursor: null_mut(),
        evacuation_limit: null_mut(),
    }));
    let href = unsafe { &mut *immix.get() };
    let join_data = JoinData::new();
    let mut mutator = MutatorRef::new(Mutator::new(
        immix.clone(),
        &href.safepoint,
        join_data.internal.clone(),
    ));
    href.mutators.push(&mut *mutator);
    href.safepoint
        .n_mutators
        .fetch_add(1, atomic::Ordering::Relaxed);
    mutator.stack_bounds = StackBounds::current_thread_stack_bounds();
    mutator.state_set(ThreadState::Safe, ThreadState::Unsafe);

    mutator
}

impl<Decoder: StackValueDecoder> StickyImmix<Decoder> {
    fn prepare_evacuation(&mut self) {
        for block in self.evacuation_candidates.drain(..) {
            unsafe {
                (*block).set_evacuation_candidate(false);
            }
        }
        self.evacuation_cursor = null_mut();
        self.evacuation_limit = null_mut();
        if !self.evacuation_enabled {
            return;
        }

        let mut to_space_bytes = self
            .space
            .free_blocks
            .len()
            .saturating_mul(IMMIX_BLOCK_SIZE - IMMIX_LINE_SIZE);
        if to_space_bytes == 0 {
            return;
        }

        for block in self.space.reusable_blocks.iter() {
            let live_lines = match unsafe { (*block).state() } {
                BlockState::Reusable { unavailable_lines } => unavailable_lines as usize,
                _ => continue,
            };
            if live_lines > IMMIX_LINES_PER_BLOCK / 2 {
                continue;
            }

            let live_bytes = live_lines * IMMIX_LINE_SIZE;
            if live_bytes > to_space_bytes {
                continue;
            }

            unsafe {
                (*block).set_evacuation_candidate(true);
                (*block).set_pinned_for_evacuation(false);
            }
            self.evacuation_candidates.push(block);
            to_space_bytes -= live_bytes;
        }
    }

    unsafe fn allocate_evacuation_copy(&mut self, size: usize) -> *mut HeapObjectHeader {
        let size = align_usize(size, 8);
        if size == 0 || size > IMMIX_BLOCK_SIZE - IMMIX_LINE_SIZE {
            return null_mut();
        }

        if self.evacuation_cursor.is_null()
            || self.evacuation_cursor.add(size) > self.evacuation_limit
        {
            let block = self.space.get_clean_block();
            if block.is_null() {
                return null_mut();
            }
            self.evacuation_cursor = (*block).start_address();
            self.evacuation_limit = (*block).end_address();
        }

        let copy = self.evacuation_cursor;
        self.evacuation_cursor = self.evacuation_cursor.add(size);
        self.space.mark_bitmap.set(copy);
        copy.cast()
    }

    unsafe fn after_mark_constraints(&mut self) {
        let this = self as *mut Self;
        (*this).constraints.retain_mut(|constraint| {
            if constraint.is_over() {
                false
            } else {
                if constraint.runs_at() == MarkingConstraintRuns::AfterMark {
                    constraint.run(self);
                }
                true
            }
        });
    }
    unsafe fn before_mark_constraints(&mut self) {
        let this = self as *mut Self;
        (*this).constraints.retain_mut(|constraint| {
            if constraint.is_over() {
                false
            } else {
                if constraint.runs_at() == MarkingConstraintRuns::BeforeMark {
                    constraint.run(self);
                }
                true
            }
        });
    }
    #[cold]
    unsafe fn collect_and_alloc<T: Collectable + Sized + 'static>(
        &mut self,
        mutator: &mut MutatorRef<Self>,

        mut value: T,
    ) -> Gc<T, Self> {
        self.collect_alloc_failure(mutator, &mut [&mut value]);

        mutator.tlab.emergency_collection = true;
        let value = self.alloc_inline(mutator, value, AllocationSpace::New);
        mutator.tlab.emergency_collection = false;
        value
    }

    #[cold]
    unsafe fn collect_and_alloc_raw(
        &mut self,
        mutator: &mut MutatorRef<Self>,

        size: usize,
        vtable: usize,
        type_id: TypeId,
    ) -> *mut HeapObjectHeader {
        self.collect_alloc_failure(mutator, &mut []);

        mutator.tlab.emergency_collection = true;
        let value = self.allocate_raw(mutator, size, type_id, vtable);
        mutator.tlab.emergency_collection = false;
        value
    }

    unsafe fn walk_stack(&mut self, mut start: *mut *mut u8, mut end: *mut *mut u8) {
        if end < start {
            std::mem::swap(&mut start, &mut end);
        }
        let mut cursor = start;
        while cursor < end {
            let pointer = cursor.read();
            if pointer.is_null() {
                cursor = cursor.add(1);
                continue;
            }
            let pointer = Decoder::decode(pointer);
            if self.space.has_address(pointer) {
                let block = ImmixBlock::from_object(pointer);
                if self.evacuation_enabled && (*block).is_evacuation_candidate() {
                    (*block).set_pinned_for_evacuation(true);
                }
                if (*block).state != BlockState::Unallocated {
                    let header = if self.space.mark_bitmap.test(pointer) {
                        pointer.cast::<HeapObjectHeader>()
                    } else {
                        self.space
                            .mark_bitmap
                            .find_header_in_range(pointer, (*block).start_address())
                    };
                    let header_address = header as usize;
                    let pointer_address = pointer as usize;
                    if !header.is_null()
                        && ImmixBlock::from_object(header.cast()) == block
                        && self.space.mark_bitmap.test(header.cast())
                    {
                        let object_end = header_address.saturating_add((*header).size());
                        if pointer_address >= header_address && pointer_address < object_end {
                            let mut header = NonNull::new_unchecked(header);
                            if self.verbose > 1 {
                                eprintln!(
                                    "[GC] Found Immix object {:p} from conservative value {:p} at {:p}",
                                    header, pointer, cursor
                                );
                            }
                            self.mark_object(&mut header);
                            cursor = cursor.add(1);
                            continue;
                        }
                    }
                }
            }

            if let Some(mut header) = NonNull::new(self.large_space.contains(pointer)) {
                self.mark_object(&mut header);
                cursor = cursor.add(1);
                continue;
            }

            cursor = cursor.add(1);
        }
    }
}

impl<Decoder: StackValueDecoder> GcBase for StickyImmix<Decoder> {
    type TLAB = ImmixAllocator;
    const SUPPORTS_TLAB: bool = true;
    type ReadBarrier = NoReadBarrier;
    const LARGE_ALLOCATION_SIZE: usize = IMMIX_BLOCK_SIZE / 2;

    fn inline_allocation_helpers(&self) -> Self::InlineAllocationHelpers {
        NoHelp
    }

    fn add_constraint<T: MarkingConstraint + 'static>(&mut self, constraint: T) {
        self.global_lock();
        self.constraints.push(Box::new(constraint));
        self.global_unlock();
    }
    fn inspect(&self, mut f: impl FnMut(Gc<dyn Collectable, Self>) -> bool) -> bool {
        unsafe {
            self.large_space_lock.lock();
            self.large_space.allocations.iter().for_each(|alloc| {
                f(std::mem::transmute((**alloc).cell()));
            });
            self.large_space_lock.unlock();
            let start = self.immix_space().map.start();
            let end = self.immix_space().map.end();
            self.immix_space()
                .mark_bitmap
                .visit_marked_range(start, end, |ptr| {
                    f(std::mem::transmute(ptr));
                });
        }
        true
    }
    fn allocate_raw(
        &mut self,
        mutator: &mut MutatorRef<Self>,
        size: usize,
        type_id: std::any::TypeId,
        vtable: usize,
    ) -> *mut HeapObjectHeader {
        let alloc = &mut mutator.tlab;
        let size = align_usize(size + size_of::<HeapObjectHeader>(), 8);
        unsafe {
            let memory = if size >= Self::LARGE_ALLOCATION_SIZE {
                alloc.alloc(size)
            } else {
                self.large_space.allocate(size).cast::<u8>()
            };

            if memory.is_null() {
                return self.collect_and_alloc_raw(mutator, size, vtable, type_id);
            }
            let object = memory.cast::<HeapObjectHeader>();
            (*object).set_vtable(vtable);
            (*object).type_id = make_small_type_id(type_id);
            (*object).set_size(size);

            let gced: Gc<(), Self> = Gc {
                base: NonNull::new_unchecked(object),
                marker: Default::default(),
            };
            self.post_alloc(gced);
            object
        }
    }
    fn allocate_weak<T: Collectable + ?Sized>(
        &mut self,
        mutator: &mut MutatorRef<Self>,
        value: Gc<T, Self>,
    ) -> Weak<T, Self> {
        let weak_ref = unsafe { Weak::<T, Self>::create(mutator, value) };
        self.global_heap_lock.lock();
        self.weak_refs.push(weak_ref.to_dyn());
        unsafe {
            self.global_heap_lock.unlock();
        }
        weak_ref
    }
    #[inline]
    fn alloc_inline<T: Collectable + Sized + 'static>(
        &mut self,
        mutator: &mut MutatorRef<Self>,
        value: T,
        _space: AllocationSpace,
    ) -> Gc<T, Self> {
        let alloc = &mut mutator.tlab;
        let size = align_usize(value.allocation_size() + size_of::<HeapObjectHeader>(), 8);
        unsafe {
            let memory = alloc.alloc(size);

            if memory.is_null() {
                return self.collect_and_alloc(mutator, value);
            }
            let object = memory.cast::<HeapObjectHeader>();
            (*object).set_metadata(vtable_of::<T>());
            (*object).type_id = small_type_id::<T>();
            (*object).set_size(size);
            ((*object).data() as *mut T).write(value);
            let gced = Gc {
                base: NonNull::new_unchecked(object),
                marker: Default::default(),
            };
            self.post_alloc(gced);
            gced
        }
    }

    fn collect(&mut self, mutator: &mut MutatorRef<Self>, keep: &mut [&mut dyn Trace]) {
        match SafepointScope::new(mutator.clone()) {
            Some(safepoint) => unsafe {
                mutator.last_sp.set(approximate_stack_pointer());
                let time = if self.verbose > 0 {
                    Some(std::time::Instant::now())
                } else {
                    None
                };

                self.global_heap_lock.lock();
                self.large_space_lock.lock();
                let mark_phase = std::time::Instant::now();
                self.prepare_evacuation();
                self.large_space.prepare_for_marking(false);
                self.large_space.prepare_for_conservative_scan();
                self.space.prepare(true);
                for i in 0..self.mutators.len() {
                    let mutator = self.mutators[i];
                    (*mutator).reset_tlab();
                    self.walk_stack(
                        (*mutator).stack_bounds.origin.cast(),
                        (*mutator).last_sp.get().cast(),
                    );
                    (*mutator).shadow_stack().walk(|entry| {
                        entry.trace(self);
                    });
                }
                self.before_mark_constraints();
                for object in keep {
                    object.trace(self);
                }

                while let Some(object) = self.mark_stack.pop() {
                    (*object).get_dyn().trace(self);
                }
                self.after_mark_constraints();
                let mark_phase = mark_phase.elapsed();
                let prev =
                    self.space.num_bytes_allocated.load(Ordering::Relaxed) + self.large_space.bytes;
                self.space.num_bytes_allocated.store(0, Ordering::Relaxed);
                let mark_color = self.mark_color;
                let sweep_phase = std::time::Instant::now();
                self.weak_refs.retain_mut(|object| {
                    let mut header = object.base();
                    if (*header).is_forwarded() {
                        header = (*header).vtable() as *mut HeapObjectHeader;
                        object.set_base(header);
                    }
                    if (*header).get_color() == mark_color {
                        object.after_mark(|header| {
                            let header = if (*header).is_forwarded() {
                                (*header).vtable() as *mut HeapObjectHeader
                            } else {
                                header
                            };
                            if (*header).get_color() == mark_color {
                                header
                            } else {
                                null_mut()
                            }
                        });
                        true
                    } else {
                        false
                    }
                });

                self.large_space.sweep();
                self.large_space.prepare_for_allocation(false);
                self.space.release(self.alloc_color);
                self.evacuation_cursor = null_mut();
                self.evacuation_limit = null_mut();

                let bytes_allocated =
                    self.space.num_bytes_allocated.load(Ordering::Relaxed) + self.large_space.bytes;
                let target_size = self
                    .space
                    .min_heap_size
                    .max((bytes_allocated as f64 * self.growth_multiplier) as usize)
                    .min(self.space.max_heap_size);

                self.space
                    .target_footprint
                    .store(target_size, Ordering::Relaxed);
                let sweep_phase = sweep_phase.elapsed();
                if let Some(time) = time {
                    let elapsed = time.elapsed();
                    eprintln!(
                        "[gc] GC({}) Pause Sticky Immix collection {}->{}({}) {:.4}ms (mark {:.4}ms, sweep {:.4}ms)",
                        self.total_gcs,
                        formatted_size(prev),
                        formatted_size(bytes_allocated),
                        formatted_size(target_size),
                        elapsed.as_micros() as f64 / 1000.0,
                        mark_phase.as_micros() as f64 / 1000.0,
                        sweep_phase.as_micros() as f64 / 1000.0
                    );
                }
                self.total_gcs += 1;
                std::mem::swap(&mut self.alloc_color, &mut self.mark_color);
                drop(safepoint);

                self.global_heap_lock.unlock();
                self.large_space_lock.unlock();
            },
            None => return,
        }
    }

    fn alloc_tlab_area(&mut self, _mutator: &MutatorRef<Self>, _size: usize) -> *mut u8 {
        null_mut()
    }
    fn safepoint(&self) -> &GlobalSafepoint {
        &self.safepoint
    }

    fn attach_current_thread(&mut self, mutator: *mut Mutator<Self>) {
        self.global_heap_lock.lock();
        self.safepoint.n_mutators.fetch_add(1, Ordering::Relaxed);
        self.mutators.push(mutator);
        unsafe { self.global_heap_lock.unlock() };
    }

    fn detach_current_thread(&mut self, mutator: *mut Mutator<Self>) {
        self.global_heap_lock.lock();

        let mut detached = false;
        self.mutators.retain(|x| {
            let x = *x;
            let y = mutator;
            if x == y {
                detached = true;
                false
            } else {
                true
            }
        });
        self.safepoint.n_mutators.fetch_sub(1, Ordering::Relaxed);
        assert!(detached, "mutator must be detached");
        unsafe {
            self.global_heap_lock.unlock();
        }
    }

    fn global_lock(&self) {
        self.global_heap_lock.lock();
    }
    fn global_unlock(&self) {
        unsafe {
            debug_assert!(self.global_heap_lock.is_locked());
            self.global_heap_lock.unlock();
        }
    }

    fn mutators(&self) -> &[*mut Mutator<Self>] {
        assert!(self.global_heap_lock.is_locked());
        &self.mutators
    }

    fn allocate_large<T: Collectable + Sized + 'static>(
        &mut self,
        _mutator: &mut MutatorRef<Self>,
        value: T,
    ) -> crate::api::Gc<T, Self> {
        unsafe {
            let size = value.allocation_size() + size_of::<HeapObjectHeader>();
            self.large_space_lock.lock();
            let object = self.large_space.allocate(size);
            (*object).set_metadata(vtable_of::<T>());
            (*object).type_id = small_type_id::<T>();
            let gc = Gc {
                base: NonNull::new_unchecked(object),
                marker: PhantomData,
            };
            ((*object).data() as *mut T).write(value);
            self.large_space_lock.unlock();
            self.post_alloc(gc);
            gc
        }
    }
    #[inline(always)]
    fn post_alloc<T: Collectable + Sized + 'static>(&mut self, value: Gc<T, Self>) {
        unsafe {
            let base = value.base.as_ptr();
            (*base).force_set_color(self.alloc_color);
            //self.space.mark_bitmap.set<true>(base as _);
        }
    }
}

impl<Decoder: StackValueDecoder> Visitor for StickyImmix<Decoder> {
    fn mark_object(&mut self, root: &mut NonNull<HeapObjectHeader>) {
        unsafe {
            let mut object = root.as_ptr();
            if (*object).is_forwarded() {
                object = (*object).vtable() as *mut HeapObjectHeader;
                *root = NonNull::new_unchecked(object);
            }
            if !(*object).set_color(self.alloc_color, self.mark_color) {
                if self.space.has_address(object.cast()) {
                    let block = ImmixBlock::from_object(object.cast());
                    if self.evacuation_enabled
                        && (*block).is_evacuation_candidate()
                        && !(*block).is_pinned_for_evacuation()
                    {
                        let size = (*object).size();
                        let copy = self.allocate_evacuation_copy(size);
                        if !copy.is_null() {
                            std::ptr::copy_nonoverlapping(object.cast::<u8>(), copy.cast(), size);
                            (*copy).force_set_color(self.mark_color);
                            (*object).set_forwarded(copy as usize);
                            self.space.mark_bitmap.clear(object.cast());
                            *root = NonNull::new_unchecked(copy);
                            object = copy;
                        }
                    }
                    self.space.mark_lines(object);
                } else {
                    (*PreciseAllocation::from_cell(object)).test_and_set_marked();
                }
                self.mark_stack.push(object);
            }
        }
    }
}

/*
pub type Histogram = [usize; Defrag::NUM_BINS];

pub struct Defrag {
    in_defrag_collection: AtomicBool,
    defrag_space_exhausted: AtomicBool,
    pub mark_histograms: Mutex<Vec<Histogram>>,
    /// A block with number of holes greater than this threshold will be defragmented.
    pub defrag_spill_threshold: AtomicUsize,
    /// The number of remaining clean pages in defrag space.
    available_clean_pages_for_defrag: AtomicUsize,
}
impl Defrag {
    const NUM_BINS: usize = (IMMIX_LINES_PER_BLOCK >> 1) + 1;
    const DEFRAG_LINE_REUSE_RATIO: f32 = 0.99;
    const MIN_SPILL_THRESHOLD: usize = 2;
    const DEFRAG_STRESS: bool = false;
    const DEFRAG_HEADROOM_PERCENT: usize = 2;

    /// Allocate a new local histogram.
    pub const fn new_histogram(&self) -> Histogram {
        [0; Self::NUM_BINS]
    }

    /// Report back a completed mark histogram
    #[inline(always)]
    pub fn add_completed_mark_histogram(&self, histogram: Histogram) {
        self.mark_histograms.lock().push(histogram)
    }

    /// Check if the current GC is a defrag GC.
    #[inline(always)]
    pub fn in_defrag(&self) -> bool {
        self.in_defrag_collection.load(Ordering::Acquire)
    }

    /// Determine whether the current GC should do defragmentation.
    pub fn decide_whether_to_defrag(
        &self,
        emergency_collection: bool,
        collect_whole_heap: bool,
        collection_attempts: usize,
        user_triggered: bool,
        exhausted_reusable_space: bool,
        full_heap_system_gc: bool,
    ) {
        let in_defrag = true
            && (emergency_collection
                || (collection_attempts > 1)
                || !exhausted_reusable_space
                || Self::DEFRAG_STRESS
                || (collect_whole_heap && user_triggered && full_heap_system_gc));
        // println!("Defrag: {}", in_defrag);
        self.in_defrag_collection
            .store(in_defrag, Ordering::Release)
    }

    /// Get the number of defrag headroom pages.
    pub fn defrag_headroom_pages(&self, space: &ImmixSpace) -> usize {
        space.get_reserved_pages() * Self::DEFRAG_HEADROOM_PERCENT / 100
    }

    /// Check if the defrag space is exhausted.
    #[inline(always)]
    pub fn space_exhausted(&self) -> bool {
        self.defrag_space_exhausted.load(Ordering::Acquire)
    }

    /// Update available_clean_pages_for_defrag counter when a clean block is allocated.
    pub fn notify_new_clean_block(&self, copy: bool) {
        if copy {
            let available_clean_pages_for_defrag =
                self.available_clean_pages_for_defrag.fetch_update(
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                    |available_clean_pages_for_defrag| {
                        if available_clean_pages_for_defrag <= IMMIX_BLOCK_SIZE / PAGE_SIZE {
                            Some(0)
                        } else {
                            Some(available_clean_pages_for_defrag - (IMMIX_BLOCK_SIZE / PAGE_SIZE))
                        }
                    },
                );
            if available_clean_pages_for_defrag.unwrap() <= IMMIX_BLOCK_SIZE / PAGE_SIZE {
                self.defrag_space_exhausted.store(true, Ordering::SeqCst);
            }
        }
    }

    /// Prepare work. Should be called in ImmixSpace::prepare.
    #[allow(clippy::assertions_on_constants)]
    pub fn prepare(&self, space: &ImmixSpace<VM>) {
        self.defrag_space_exhausted.store(false, Ordering::Release);

        // Calculate available free space for defragmentation.

        let mut available_clean_pages_for_defrag = VM::VMActivePlan::global().get_total_pages()
            as isize
            - VM::VMActivePlan::global().get_pages_reserved() as isize
            + self.defrag_headroom_pages(space) as isize;
        if available_clean_pages_for_defrag < 0 {
            available_clean_pages_for_defrag = 0
        };

        self.available_clean_pages_for_defrag
            .store(available_clean_pages_for_defrag as usize, Ordering::Release);

        if self.in_defrag() {
            self.establish_defrag_spill_threshold(space)
        }

        self.available_clean_pages_for_defrag.store(
            available_clean_pages_for_defrag as usize
                + VM::VMActivePlan::global().get_collection_reserve(),
            Ordering::Release,
        );
    }

    /// Get the numebr of all the recyclable lines in all the reusable blocks.
    fn get_available_lines(
        &self,
        space: &ImmixSpace<VM>,
        spill_avail_histograms: &mut Histogram,
    ) -> usize {
        let mut total_available_lines = 0;
        for block in space.reusable_blocks.get_blocks().iter() {
            let bucket = block.get_holes();
            let unavailable_lines = match block.get_state() {
                BlockState::Reusable { unavailable_lines } => unavailable_lines as usize,
                s => unreachable!("{:?} {:?}", block, s),
            };
            let available_lines = Block::LINES - unavailable_lines;
            spill_avail_histograms[bucket] += available_lines;
            total_available_lines += available_lines;
        }
        total_available_lines
    }

    /// Calculate the defrag threshold.
    fn establish_defrag_spill_threshold(&self, space: &ImmixSpace) {
        let mut spill_avail_histograms = self.new_histogram();
        let clean_lines = self.get_available_lines(space, &mut spill_avail_histograms);
        let available_lines = clean_lines
            + (self
                .available_clean_pages_for_defrag
                .load(Ordering::Acquire)
                << (12 as usize - 8));

        // Number of lines we will evacuate.
        let mut required_lines = 0isize;
        // Number of to-space free lines we can use for defragmentation.
        let mut limit = (available_lines as f32 / Self::DEFRAG_LINE_REUSE_RATIO) as isize;
        let mut threshold = IMMIX_LINES_PER_BLOCK >> 1;
        let mark_histograms = self.mark_histograms.lock();
        // Blocks are grouped by buckets, indexed by the number of holes in the block.
        // `mark_histograms` remembers the number of live lines for each bucket.
        // Here, reversely iterate all the bucket to find a threshold that all buckets above this
        // threshold can be evacuated, without causing to-space overflow.
        for index in (Self::MIN_SPILL_THRESHOLD..Self::NUM_BINS).rev() {
            threshold = index;
            // Calculate total number of live lines in this bucket.
            let this_bucket_mark = mark_histograms
                .iter()
                .map(|v| v[threshold] as isize)
                .sum::<isize>();
            // Calculate the number of free lines in this bucket.
            let this_bucket_avail = spill_avail_histograms[threshold] as isize;
            // Update counters
            limit -= this_bucket_avail as isize;
            required_lines += this_bucket_mark;
            // Stop scanning. Lines to evacuate exceeds the free to-space lines.
            if limit < required_lines {
                break;
            }
        }
        // println!("threshold: {}", threshold);
        debug_assert!(threshold >= Self::MIN_SPILL_THRESHOLD);
        self.defrag_spill_threshold
            .store(threshold, Ordering::Release);
    }

    /// Release work. Should be called in ImmixSpace::release.
    #[allow(clippy::assertions_on_constants)]
    pub fn release(&self, _space: &ImmixSpace) {
        self.in_defrag_collection.store(false, Ordering::Release);
    }
}*/

impl<Decoder: StackValueDecoder> Drop for StickyImmix<Decoder> {
    fn drop(&mut self) {
        unsafe {
            if self.verbose > 0 {
                eprintln!("Dispose immix space at {:p}", self);
            }
            Box::from_raw(self.space as *const ImmixSpace as *mut ImmixSpace);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gc_base::NoOpStackDecoder;

    fn collector() -> MutatorRef<StickyImmix<NoOpStackDecoder>> {
        let heap_size = 4 * 1024 * 1024;
        instantiate_sticky_immix(
            StickyImmixOptions::default()
                .with_heap_size(heap_size)
                .with_initial_size(heap_size)
                .with_min_heap_size(heap_size)
                .with_max_heap_size(heap_size)
                .with_evacuation(true),
        )
    }

    fn make_evacuation_candidate<T: Collectable>(
        heap: &mut StickyImmix<NoOpStackDecoder>,
        object: &Gc<T, StickyImmix<NoOpStackDecoder>>,
    ) -> *mut ImmixBlock {
        let block = ImmixBlock::from_object(object.base.as_ptr().cast());
        unsafe {
            (*block).set_state(BlockState::Reusable {
                unavailable_lines: 1,
            });
        }
        heap.space.reusable_blocks.push(block);
        heap.prepare_evacuation();
        heap.space.prepare(true);
        block
    }

    struct Link {
        value: u64,
        next: Option<Gc<Link, StickyImmix<NoOpStackDecoder>>>,
    }

    struct MediumObject([u64; 64]);

    unsafe impl Trace for Link {
        fn trace(&mut self, visitor: &mut dyn Visitor) {
            self.next.trace(visitor);
        }
    }

    unsafe impl Trace for MediumObject {}

    unsafe impl crate::api::Finalize for Link {}
    unsafe impl crate::api::Finalize for MediumObject {}
    impl Collectable for Link {}
    impl Collectable for MediumObject {}

    #[test]
    fn evacuation_rewrites_traced_references_and_preserves_graph() {
        let mut mutator = collector();
        let child = mutator.allocate(
            Link {
                value: 42,
                next: None,
            },
            AllocationSpace::New,
        );
        let child_address = child.base.as_ptr();
        let mut root = mutator.allocate(
            Link {
                value: 7,
                next: Some(child),
            },
            AllocationSpace::New,
        );
        let root_address = root.base.as_ptr();

        let heap = mutator.heap_ref();
        let source_block = make_evacuation_candidate(heap, &root);
        heap.mark_object(&mut root.base);
        while let Some(object) = heap.mark_stack.pop() {
            unsafe {
                (*object).get_dyn().trace(heap);
            }
        }

        assert_ne!(root.base.as_ptr(), root_address);
        assert_ne!(
            ImmixBlock::from_object(root.base.as_ptr().cast()),
            source_block
        );
        assert_eq!(
            unsafe { (*root.base.as_ptr()).get_dyn().type_name() },
            std::any::type_name::<Link>()
        );
        let moved_root = unsafe { &mut *((*root.base.as_ptr()).data() as *mut Link) };
        assert_eq!(moved_root.value, 7);
        let moved_child = moved_root.next.as_ref().unwrap();
        assert_ne!(moved_child.base.as_ptr(), child_address);
        assert_eq!(
            unsafe { &*((*moved_child.base.as_ptr()).data() as *const Link) }.value,
            42
        );
        assert!(unsafe { (*root_address).is_forwarded() });
        assert!(unsafe { (*child_address).is_forwarded() });
        assert!(!heap.space.mark_bitmap.test(root_address.cast()));
        assert!(!heap.space.mark_bitmap.test(child_address.cast()));
    }

    #[test]
    fn conservative_stack_observation_pins_candidate_block() {
        let mut mutator = collector();
        let object = mutator.allocate(42_u64, AllocationSpace::New);

        let heap = mutator.heap_ref();
        let block = make_evacuation_candidate(heap, &object);
        let mut stack_slot = [object.base.as_ptr().cast::<u8>()];
        unsafe {
            heap.walk_stack(stack_slot.as_mut_ptr(), stack_slot.as_mut_ptr().add(1));
        }

        assert!(unsafe { (*block).is_pinned_for_evacuation() });
        assert_eq!(object.base.as_ptr(), stack_slot[0].cast());
        assert_eq!(unsafe { *(*object.base.as_ptr()).data().cast::<u64>() }, 42);
    }

    #[test]
    fn conservative_interior_pointer_marks_and_pins_its_object() {
        let mut mutator = collector();
        let object = mutator.allocate(
            Link {
                value: 42,
                next: None,
            },
            AllocationSpace::New,
        );
        let object_address = object.base.as_ptr();

        let heap = mutator.heap_ref();
        let block = make_evacuation_candidate(heap, &object);
        let interior = unsafe {
            (*object_address)
                .data()
                .add(std::mem::size_of::<u64>())
                .cast_mut()
        };
        let mut stack_slot = [interior];
        unsafe {
            heap.walk_stack(stack_slot.as_mut_ptr(), stack_slot.as_mut_ptr().add(1));
        }

        assert!(unsafe { (*block).is_pinned_for_evacuation() });
        assert_eq!(heap.mark_stack.as_slice(), &[object_address]);
        assert_eq!(
            unsafe { &*((*object_address).data() as *const Link) }.value,
            42
        );
    }

    #[test]
    fn marking_large_object_sets_its_line_range() {
        let mut mutator = collector();
        let object = mutator.allocate(42_u64, AllocationSpace::New);
        let block = ImmixBlock::from_object(object.base.as_ptr().cast());
        let first_line = unsafe { (*block).start_address() };
        let end_line = unsafe { first_line.add(64 * IMMIX_LINE_SIZE) };
        let chunk = unsafe { (*block).chunk() };
        let line_table = unsafe { (*chunk).line_mark_table() };
        line_table.set_range(first_line, end_line);
        let mut line = first_line;
        while line < end_line {
            assert!(line_table.test(line));
            line = unsafe { line.add(IMMIX_LINE_SIZE) };
        }
    }

    #[test]
    fn tlab_refills_and_allocates_small_and_multiline_objects() {
        let mut mutator = collector();
        assert!(StickyImmix::<NoOpStackDecoder>::SUPPORTS_TLAB);
        let small = mutator.allocate(7_u64, AllocationSpace::New);
        let multiline = mutator.allocate(MediumObject([9_u64; 64]), AllocationSpace::New);

        assert_eq!(unsafe { *(*small.base.as_ptr()).data().cast::<u64>() }, 7);
        assert_eq!(
            unsafe {
                (*multiline.base.as_ptr())
                    .data()
                    .cast::<MediumObject>()
                    .as_ref()
                    .unwrap()
                    .0[0]
            },
            9
        );
        assert!(!mutator.tlab.cursor.is_null());
        assert!(!mutator.tlab.large_cursor.is_null());
        assert!(mutator
            .heap_ref()
            .space
            .mark_bitmap
            .test(small.base.as_ptr().cast()));
        assert!(mutator
            .heap_ref()
            .space
            .mark_bitmap
            .test(multiline.base.as_ptr().cast()));
    }
}

use comet::{
    api::{Collectable, Finalize, Gc, Trace},
    gc_base::{AllocationSpace, NoOpStackDecoder},
    letroot,
    sticky_immix::{instantiate_sticky_immix, StickyImmix, StickyImmixOptions},
};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

struct Payload(u64);

unsafe impl Trace for Payload {}
unsafe impl Finalize for Payload {}
impl Collectable for Payload {}

const ROUNDS: usize = 5;
const ALLOCATIONS_PER_ROUND: usize = 100_000;
const LIVE_OBJECTS: usize = 4_096;
const COLLECTIONS_PER_ROUND: usize = 20;

fn new_mutator() -> comet::mutator::MutatorRef<StickyImmix<NoOpStackDecoder>> {
    let heap_size = 16 * 1024 * 1024;
    instantiate_sticky_immix(
        StickyImmixOptions::default()
            .with_heap_size(heap_size)
            .with_initial_size(heap_size)
            .with_min_heap_size(heap_size)
            .with_max_heap_size(heap_size),
    )
}

fn allocation_round(space: AllocationSpace) -> Duration {
    let mut mutator = new_mutator();
    let stack = mutator.shadow_stack();
    letroot!(
        last = stack,
        None::<Gc<Payload, StickyImmix<NoOpStackDecoder>>>
    );
    let start = Instant::now();

    for value in 0..ALLOCATIONS_PER_ROUND {
        *last = Some(mutator.allocate(Payload(value as u64), space));
        black_box(last.as_ref().unwrap().0);
    }

    start.elapsed()
}

fn collection_round() -> Duration {
    let mut mutator = new_mutator();
    let stack = mutator.shadow_stack();
    letroot!(
        roots = stack,
        Vec::<Gc<Payload, StickyImmix<NoOpStackDecoder>>>::new()
    );
    for value in 0..LIVE_OBJECTS {
        roots.push(mutator.allocate(Payload(value as u64), AllocationSpace::New));
    }

    let start = Instant::now();
    for _ in 0..COLLECTIONS_PER_ROUND {
        mutator.full_collection(&mut [&mut *roots]);
    }
    black_box(roots.len());
    start.elapsed()
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn main() {
    for (label, space) in [
        ("TLAB", AllocationSpace::New),
        ("inline baseline", AllocationSpace::Old),
    ] {
        let allocation = median((0..ROUNDS).map(|_| allocation_round(space)).collect());
        let allocation_ns = allocation.as_nanos() as f64 / ALLOCATIONS_PER_ROUND as f64;
        println!(
            "Sticky Immix allocation ({label}): {allocation_ns:.2} ns/object ({:.0} objects/s, {ALLOCATIONS_PER_ROUND} allocations/round, median of {ROUNDS})",
            1_000_000_000.0 / allocation_ns
        );
    }

    let collection = median((0..ROUNDS).map(|_| collection_round()).collect());
    let collection_ns = collection.as_nanos() as f64 / COLLECTIONS_PER_ROUND as f64;
    println!(
        "Sticky Immix full collection: {collection_ns:.0} ns/collection ({LIVE_OBJECTS} live objects, median of {ROUNDS})"
    );
}

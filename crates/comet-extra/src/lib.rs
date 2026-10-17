//! # comet-extra
//!
//! Collection of types that can be used with Comet.

pub mod alloc;

pub use comet::*;

#[cfg(test)]
pub(crate) fn create_heap_for_tests() -> mutator::MutatorRef<immix::Immix> {
    use comet::immix::ImmixOptions;

    immix::instantiate_immix(ImmixOptions::default().with_verbose(1))
}

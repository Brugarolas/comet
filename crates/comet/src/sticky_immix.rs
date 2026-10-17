//! Sticky Immix collector and its public construction API.
//!
//! Collections trace the full heap because Sticky Immix currently has no
//! generational write barrier. Sparse-block evacuation can be enabled
//! independently; conservative stack observations pin their containing block
//! for that collection.

pub use crate::immix::{instantiate_sticky_immix, StickyImmix, StickyImmixOptions};

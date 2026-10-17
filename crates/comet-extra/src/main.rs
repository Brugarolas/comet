use comet_extra::alloc::hash::HashMap;

fn main() {
    let options = comet::immix::ImmixOptions::default()
        .with_heap_size(128 * 1024 * 1024)
        .with_initial_size(32 * 1024 * 1024)
        .with_min_heap_size(4 * 1024 * 1024)
        .with_max_heap_size(128 * 1024 * 1024)
        .with_verbose(1);
    let mut heap = comet::immix::instantiate_immix::<comet::gc_base::NoOpStackDecoder>(options);

    let mut map = HashMap::new(&mut heap);
    map.insert(&mut heap, 0i32, 1i32);
    map.insert(&mut heap, 1, 2);
    map.insert(&mut heap, 2, 3);
    println!("{}", map.get(&2).unwrap());
    for (_, val) in map.iter_mut() {
        *val += 42;
    }
    for (key, val) in map.iter() {
        println!("{}->{}", key, val);
    }
}

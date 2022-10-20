extern crate chsim;


use ndarray::{
    Array1
    , Array2
    , Axis
};


fn main() {
    let data_len=1<<18;
    let nch=data_len/2;
    let mut outfile=ndarray_npy::NpzWriter::new(std::fs::File::create("a.npz").unwrap());
    let fg=chsim::fg_spec(data_len/2, 50e6, 200e6, &|_f|{30.0});
    let eor=chsim::realize_eor_spec(&|i| {if i==0{0.0} else{(i as f64).powf(-1.0)}}, nch, 50e6, 200e6, &mut rand::thread_rng(),0.1);
    //let eor:Vec<_>=(0..nch).map(|c| ((c as f64)/1024 as f64*2.0*3.14159).sin()*1.0).collect();
    
    //let total_signal=chsim::simulate_time_series(data_len, 50e6, 200e6,& |f| {806.0*(f/100e6).powf(-2.7)},& |n| if n>0 {(n as f64).powf(-1.0)} else {0.0}, 0.0, &mut rand::thread_rng());
    assert!(fg.len()==eor.len());
    let mut total_spec:Vec<_>=fg.iter().zip(eor.iter()).map(|(a,b)| a+b).collect();
    for x in &mut total_spec{
        if *x<0.0{
            *x=0.0;
        }
    }

    let nreal=512;
    let mut result=Array2::<f64>::zeros((nreal, data_len));

    result.axis_iter_mut(Axis(0)).enumerate().for_each(|(i, mut c)|{
        println!("{}", i);
        let total_signal=chsim::simulate_time_series(&total_spec, &mut rand::thread_rng());
        let total_signal=Array1::from(total_signal);
        c.assign(&total_signal);
    });
    outfile.add_array("voltage", &result).unwrap();
}

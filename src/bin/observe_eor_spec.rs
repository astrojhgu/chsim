use ndarray::{Array1, Array2, ArrayView1, Axis, s};
use num_complex::Complex;
use num_traits::FloatConst;
use rspfb::{csp_pfb::CspPfb, cspfb, oscillator::COscillator, ospfb, windowed_fir::coeff};
use itertools_num::linspace;

use ndarray_npy::NpzWriter;



pub fn main(){
    let nch_coarse = 1024;
    //let coeff_coarse = coeff::<f64>(nch_coarse / 2, 16, 1.1);//best
    let coeff_coarse = coeff::<f64>(nch_coarse / 2, 12, 1.3);//bad

    let nch_fine = 16;
    let coeff_fine = coeff::<f64>(nch_fine*2, 2, 0.5);
    let selected_ch: Vec<_> = (0..nch_coarse/2+1).collect();

    let data_len=1<<22;
    let nch=data_len/2;
    let mut outfile=ndarray_npy::NpzWriter::new(std::fs::File::create("a.npz").unwrap());
    let fg=chsim::fg_spec(data_len/2, 50e6, 200e6, &|f|{3.0});



    let eor=chsim::realize_eor_spec(&|i| {if i==0{0.0} else{(i as f64).powf(-1.0)}}, nch, 50e6, 200e6, &mut rand::thread_rng(),0.1);
    //let eor:Vec<_>=(0..nch).map(|c| ((c as f64)/4096 as f64*2.0*3.14159).sin()*1.0).collect();
    
    //let total_signal=chsim::simulate_time_series(data_len, 50e6, 200e6,& |f| {806.0*(f/100e6).powf(-2.7)},& |n| if n>0 {(n as f64).powf(-1.0)} else {0.0}, 0.0, &mut rand::thread_rng());

    
    assert!(fg.len()==eor.len());
    let mut total_spec:Vec<f64>=fg.iter().zip(eor.iter()).map(|(a,b)| a+b).collect();
    for x in &mut total_spec{
        if *x<0.0{
            *x=0.0;
        }
    }
    
    let mut spec=Array1::from(vec![0.0; nch_coarse*nch_fine/2]);
    let freq_raw=Array1::from(linspace(0.0, 200e6, data_len/2).collect::<Vec<_>>());
    let freq_obs=Array1::from(linspace(0.0, 200e6, spec.len()).collect::<Vec<_>>());
    

    outfile.add_array("input_spec", &ArrayView1::from(&total_spec));
    outfile.add_array("freq_raw", &freq_raw);
    outfile.add_array("freq_obs", &freq_obs);
    //////////
    
    
    for i in 0..100{
        println!("{}", i);
        let total_signal:Vec<_>=chsim::simulate_time_series(&total_spec, &mut rand::thread_rng()).iter().map(|x| Complex::<f64>::from(x)).collect();
        let mut channelizer_coarse = ospfb::Analyzer::<Complex<f64>, f64>::new(nch_coarse, ArrayView1::from(&coeff_coarse));
        let channelizer_fine =
            cspfb::Analyzer::<Complex<f64>, f64>::new(nch_fine * 2, ArrayView1::from(&coeff_fine));
        
        let mut csp = CspPfb::new(&selected_ch, &channelizer_fine);
    
        let mut station_output = channelizer_coarse.analyze_par(&total_signal);
        let fine_channels:Array2<_> = csp.analyze_par(station_output.view_mut()).slice(s![8..8192+8, 2..]).to_owned();
        //let spec1=fine_channels.sum_axis(Axis(1));
        let spec1=fine_channels.map(|x| x.norm_sqr()).sum_axis(Axis(1));
        spec+=&spec1;    
    }
    
   outfile.add_array("spec", &spec);
}
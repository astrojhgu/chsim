pub mod utils;

use rand::{
    Rng
};

use rand_distr::{
    Normal
};

use num_traits::{
    Float
};

use num_complex::{
    Complex
};

use rustfft::{
    FftPlanner
};

pub fn stddev(x:&[f64])->f64{
    let xmean:f64=x.iter().sum::<f64>()/x.len() as f64;
    let x2mean:f64=x.iter().map(|x|x.powi(2)).sum::<f64>()/x.len() as f64;
    (x2mean-xmean.powi(2)).sqrt()
}

pub fn realize_eor_spec<R>(eor_ps: &dyn Fn(usize)->f64, nch: usize, fmin: f64, fmax: f64, rng: &mut R, norm:f64)->Vec<f64>
where R: Rng
//calculate radio spectrum according to matter ps
{
    let mut plan=FftPlanner::<f64>::new();
    let fft=plan.plan_fft_inverse(nch);

    let white:Vec<_>=rng.sample_iter(Normal::<f64>::new(0.0, 1.0/2_f64.sqrt()).unwrap()).take(nch*2).collect();
    let mut white:Vec<_>=white.chunks(2).map(|x| Complex::new(x[0],x[1])).enumerate().map(|(i,x)| {
        x*eor_ps(i).sqrt()
    }).collect();

    utils::make_real_ps(&mut white);
    fft.process(&mut white);
    let result:Vec<_>=white.iter().map(|x| x.re/(nch as f64).sqrt()).enumerate().map(|(i, x)|{if (i as f64) < (nch as f64/fmax*fmin) {0.0} else {x}}).collect();
    let s=stddev(&result[(nch as f64/fmax*fmin) as usize ..]);
    result.iter().map(|x| x/s*norm).collect()
}

pub fn fg_spec(nch:usize, fmin: f64, fmax: f64, spec: &dyn Fn(f64)->f64)->Vec<f64>{
    (0..nch).map(|i|{ fmax/nch as f64*i as f64 }).map(|f|{if f<fmin {0.0} else{spec(f)}}).collect()
}

pub fn freq_list(nch: usize, fmin:f64, fmax:f64)->Vec<f64>{
    let df=(fmax-fmin)/nch as f64;
    (0..nch).map(|i| (i as f64+0.5)*df).collect()
}


pub fn simulate_time_series<R:Rng>(radio_spec: &[f64], rng: &mut R)->Vec<f64>{
    let nch=radio_spec.len();
    let data_len=nch*2;
    let white:Vec<_>=rng.sample_iter(Normal::<f64>::new(0.0, 1.0/2_f64.sqrt()).unwrap()).take(nch*2).collect();
    let mut white:Vec<_>=white.chunks(2).map(|x| Complex::new(x[0],x[1])).zip(radio_spec.iter()).map(|(w,s)|{w*s.sqrt()}).collect();
    white.append(&mut vec![Complex::<f64>::default(); nch]);
    utils::make_real_ps(&mut white);
    let mut plan=FftPlanner::<f64>::new();
    let fft=plan.plan_fft_inverse(nch*2);
    fft.process(&mut white);
    white.iter().map(|x| x.re/data_len as f64).collect()
}

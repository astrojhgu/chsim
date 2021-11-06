use num_complex::{
    Complex
};

use num_traits::{
    Float
};

pub fn make_real_ps<T>(x:&mut [Complex<T>])
where T:Float{
    let n=x.len();
    x[0]=Complex::<T>::from(T::zero());
    x[n/2]=Complex::<T>::from(x[n/2].norm()*x[n/2].re.signum());
    for i in 1..n/2{
        x[n-i]=x[i].conj();
    }
}
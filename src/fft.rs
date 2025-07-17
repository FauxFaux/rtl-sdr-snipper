use num_complex::Complex;
use rustfft::Fft;
use std::sync::Arc;

pub struct SimpleFft {
    inner: Arc<dyn Fft<f32>>,
    pub(crate) len: usize,
    scratch: Box<[Complex<f32>]>,
    window: Box<[f32]>,
}

impl SimpleFft {
    pub fn new(fft_width: usize) -> Self {
        let inner = rustfft::FftPlanner::new().plan_fft_forward(fft_width);
        SimpleFft {
            len: fft_width,
            scratch: vec![Complex::new(0.0f32, 0.0); inner.get_inplace_scratch_len()]
                .into_boxed_slice(),
            window: generate_blackman_harris_window(fft_width),
            inner,
        }
    }

    #[inline]
    pub fn process(&mut self, chunk: &[u8]) -> Vec<f32> {
        assert_eq!(
            chunk.len(),
            2 * self.inner.len(),
            "chunk length must match FFT width"
        );

        let mut chunk = chunk
            .chunks_exact(2)
            .map(|pair| {
                Complex::new(
                    (f32::from(pair[0]) - 128.0) / 128.0,
                    (f32::from(pair[1]) - 128.0) / 128.0,
                )
            })
            .zip(self.window.iter())
            .map(|(c, w)| c * w)
            .collect::<Vec<_>>();
        self.inner
            .process_with_scratch(&mut chunk, &mut self.scratch);
        chunk.into_iter().map(|v| v.norm()).collect::<Vec<_>>()
    }
}

fn generate_blackman_harris_window(n: usize) -> Box<[f32]> {
    let mut window = Vec::with_capacity(n);
    for i in 0..n {
        let x = std::f32::consts::TAU * i as f32 / (n - 1) as f32;
        let value =
            0.35875 - 0.48829 * x.cos() + 0.14128 * (2.0 * x).cos() - 0.01168 * (3.0 * x).cos();
        window.push(value);
    }
    window.into_boxed_slice()
}

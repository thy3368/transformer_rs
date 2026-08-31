pub(crate) fn linear(
    x: &[f32],
    rows: usize,
    input: usize,
    weight: &[f32],
    output: usize,
    bias: &[f32],
) -> Vec<f32> {
    let mut y = vec![0.0; rows * output];
    for r in 0..rows {
        for o in 0..output {
            let mut v = bias[o];
            for i in 0..input {
                v += x[r * input + i] * weight[i * output + o];
            }
            y[r * output + o] = v;
        }
    }
    y
}
pub(crate) fn layer_norm(
    x: &[f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut y = vec![0.0; x.len()];
    let mut means = vec![0.0; rows];
    let mut inv = vec![0.0; rows];
    for r in 0..rows {
        let m = x[r * cols..(r + 1) * cols].iter().sum::<f32>() / cols as f32;
        let var = x[r * cols..(r + 1) * cols]
            .iter()
            .map(|v| (v - m) * (v - m))
            .sum::<f32>()
            / cols as f32;
        let iv = (var + 1e-5).sqrt().recip();
        means[r] = m;
        inv[r] = iv;
        for c in 0..cols {
            y[r * cols + c] = (x[r * cols + c] - m) * iv * gamma[c] + beta[c];
        }
    }
    (y, means, inv)
}
pub(crate) fn linear_backward(
    x: &[f32],
    dy: &[f32],
    rows: usize,
    input: usize,
    weight: &[f32],
    output: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut dx = vec![0.0; rows * input];
    let mut dw = vec![0.0; input * output];
    let mut db = vec![0.0; output];
    for r in 0..rows {
        for o in 0..output {
            let g = dy[r * output + o];
            db[o] += g;
            for i in 0..input {
                dx[r * input + i] += g * weight[i * output + o];
                dw[i * output + o] += x[r * input + i] * g;
            }
        }
    }
    (dx, dw, db)
}
pub(crate) fn layer_norm_backward(
    x: &[f32],
    dy: &[f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    means: &[f32],
    inv: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut dx = vec![0.0; x.len()];
    let mut dg = vec![0.0; cols];
    let mut db = vec![0.0; cols];
    for r in 0..rows {
        let mut sum = 0.0;
        let mut sum_x = 0.0;
        for c in 0..cols {
            let idx = r * cols + c;
            let g = dy[idx] * gamma[c];
            sum += g;
            sum_x += g * (x[idx] - means[r]) * inv[r];
            dg[c] += dy[idx] * (x[idx] - means[r]) * inv[r];
            db[c] += dy[idx];
        }
        for c in 0..cols {
            let idx = r * cols + c;
            let g = dy[idx] * gamma[c];
            let xn = (x[idx] - means[r]) * inv[r];
            dx[idx] = inv[r] * (g - (sum + xn * sum_x) / cols as f32);
        }
    }
    (dx, dg, db)
}

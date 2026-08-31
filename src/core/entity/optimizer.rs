use super::{Gradients, TransformerModel};
pub struct Adam {
    learning_rate: f32,
    step: usize,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}
impl Adam {
    pub fn new(model: &TransformerModel, learning_rate: f32) -> Self {
        let sizes = model.parameter_sizes();
        Self {
            learning_rate,
            step: 0,
            m: sizes.iter().map(|&n| vec![0.0; n]).collect(),
            v: sizes.iter().map(|&n| vec![0.0; n]).collect(),
        }
    }
    pub fn step(&mut self, model: &mut TransformerModel, gradients: &mut Gradients, max_norm: f32) {
        let norm = gradients
            .values
            .iter()
            .flatten()
            .map(|g| g * g)
            .sum::<f32>()
            .sqrt();
        if norm > max_norm {
            let scale = max_norm / norm;
            for g in gradients.values.iter_mut().flatten() {
                *g *= scale;
            }
        }
        self.step += 1;
        model.apply(
            gradients,
            &mut self.m,
            &mut self.v,
            self.step,
            self.learning_rate,
        );
    }
}

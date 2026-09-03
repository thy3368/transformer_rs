use candle_core::{Device, Result, Tensor};
use candle_nn::{Embedding as CandleEmbedding, Module, VarBuilder};

pub struct Embedding {
    inner: CandleEmbedding,
    vocab_size: usize,
    d_model: usize,
    device: Device,
}

impl Embedding {
    pub fn new(vocab_size: usize, d_model: usize, device: &Device) -> Result<Self> {
        let weight = Tensor::randn(0f32, 1f32, (vocab_size, d_model), device)?;
        Self::from_weight(weight, vocab_size, d_model, device.clone())
    }

    pub fn from_var_builder(vocab_size: usize, d_model: usize, vb: VarBuilder) -> Result<Self> {
        let inner = candle_nn::embedding(vocab_size, d_model, vb.pp("embedding"))?;
        let device = inner.embeddings().device().clone();
        Ok(Self {
            inner,
            vocab_size,
            d_model,
            device,
        })
    }

    fn from_weight(
        weight: Tensor,
        vocab_size: usize,
        d_model: usize,
        device: Device,
    ) -> Result<Self> {
        let inner = CandleEmbedding::new(weight, d_model);
        Ok(Self {
            inner,
            vocab_size,
            d_model,
            device,
        })
    }

    fn check_ids(&self, ids: &Tensor) -> Result<()> {
        let values = ids.flatten_all()?.to_vec1::<u32>()?;
        if let Some((index, id)) = values
            .iter()
            .enumerate()
            .find(|(_, id)| **id >= self.vocab_size as u32)
        {
            candle_core::bail!(
                "embedding token id {} at index {} is out of range for vocabulary size {}",
                id,
                index,
                self.vocab_size
            )
        }
        Ok(())
    }

    pub fn forward(&self, token_ids: &[u32]) -> Result<Tensor> {
        let ids = Tensor::new(token_ids, &self.device)?;
        self.check_ids(&ids)?;
        self.inner.forward(&ids)
    }

    pub fn forward_tensor(&self, input_ids: &Tensor) -> Result<Tensor> {
        if input_ids.rank() != 2 {
            candle_core::bail!(
                "embedding input must have shape [batch, seq_len], got {:?}",
                input_ids.dims()
            );
        }
        self.check_ids(input_ids)?;
        self.inner.forward(input_ids)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    pub fn d_model(&self) -> usize {
        self.d_model
    }
}

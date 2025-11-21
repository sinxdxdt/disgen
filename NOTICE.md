# Third-Party Software Notices

This software includes or vendors the following third-party components:

## DisCoder

**Repository**: https://github.com/ETH-DISCO/discoder  
**License**: MIT License  
**Copyright**: (c) 2024 ETH DISCO  
**Vendored Location**: `disgen/vendor/discoder/`  
**Purpose**: High-fidelity neural vocoder using neural audio codecs for mel-to-waveform synthesis

DisCoder is a GAN-based encoder-decoder architecture informed by the Descript Audio Codec (DAC) to reconstruct high-fidelity 44.1 kHz audio from mel spectrograms.

**Citation**:
```bibtex
@inproceedings{discoder2025,
  title={DisCoder: High-Fidelity Music Vocoder Using Neural Audio Codecs},
  author={},
  booktitle={ICASSP 2025},
  year={2025}
}
```

**Full License Text**: See `disgen/vendor/LICENSES/DisCoder-LICENSE`

---

## License Compatibility

disgen is released under the MIT License. All vendored components are also under MIT License, ensuring full compatibility for redistribution and modification.

## Acknowledgments

We thank the ETH DISCO team for their excellent work on DisCoder and for releasing it under a permissive open-source license that allows for vendoring and integration into other projects.

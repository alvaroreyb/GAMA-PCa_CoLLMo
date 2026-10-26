# Third-party notices

The project license applies only to material the project authors are entitled to license. It does not grant rights to external clinical knowledge, guideline publications, model weights, APIs or third-party dependencies.

- The NER+Regex implementation is the authors' existing GenEHR pipeline, reused here with a configurable input path. Its six BSC/PlanTL and three MedSpaner model identifiers are listed in `docs/ner.md`; consult each model repository's terms before downloading or using weights.
- GLM 5.2 and Kimi K2.6 are external models accessed through a separately installed Ollama service. Their model and service terms apply independently.
- Python dependencies, including PyTorch, Transformers, pandas, NumPy, Pydantic, PyYAML, Requests and pytest, retain their own licenses. They are installed separately, not vendored into this repository.
- The PolyForm license text is reproduced from the PolyForm Project's official license repository, version 1.0.0: https://github.com/polyformproject/polyform-licenses/blob/1.0.0/PolyForm-Noncommercial-1.0.0.md
- The curated YAML clinical knowledge is distributed as a separate dataset with its own license; it is not bundled with this software. The original EAU guideline document is not distributed in either package.

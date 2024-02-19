# 🦙🌲🤏 IDDD-LLM

## ID Data Descriptor LLM

GPT3.5 기반으로 생성하여 검수한 report dataset 665건으로 llama2-13b 모델을 SFT(Supervised Fune-Tuning) 방식으로 학습하여 ID사업에 최적화된 언어 모델을 구축하였습니다. FT 과정에서 LoRA 외 VeRA, DoRA 등 최신 PEFT 기법을 비교, 분석하여 보다 효율적인 학습 방법을 모색하였습니다.

본 코드는 [alpaca-lora](https://github.com/tloen/alpaca-lora.git) 코드를 기반으로 하고 있습니다.
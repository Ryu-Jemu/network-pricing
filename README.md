# 5G 네트워크 슬라이싱 SLA-제약 동적 가격 최적화 (제약 강화학습)

5G 네트워크 슬라이싱(URLLC·eMBB)의 Three-Part Tariff 구조 하 동적 가격
결정을 다루는 연구 코드입니다. 두 단계의 연구를 포함합니다.

1. **학술대회 논문 (ASK2026, KIPS_C2026A0229)** — 외생 QoS 환경에서
   PPO/SAC/TD3 기반 동적 가격과 이탈 민감도(m) 분석. 본 저장소에서 전부
   재현 가능.
2. **학술지 확장 (KTCCS 투고)** — 가격→가입자→트래픽 부하→QoS 루프를
   내생화하고, SLA 준수를 제약 MDP(CMDP)로 정식화하여 PPO-Lagrangian으로
   해결. 강력 정적 베이스라인(BO 오라클·Oracle MPC)과 7시드 통계 프로토콜
   포함.

*English summary*: Research code for SLA-constrained dynamic pricing of
5G network slices under a three-part tariff. The journal extension
endogenizes QoS (price → subscribers → load → QoS) and enforces the SLA
through a constrained MDP solved with PPO-Lagrangian, evaluated against
strong static oracles (GP-UCB Bayesian optimization, oracle MPC) with a
paired 7-seed evaluation protocol.

## 설치

Python ≥ 3.10 (Windows/macOS/Linux).

```bash
pip install -r requirements.txt     # 고정 버전 (실험 재현용)
# 또는
pip install -e .                    # 범위 버전
```

## 테스트

```bash
python -m pytest tests/ -q
```

환경 회귀(발표 논문 환경과의 비트단위 일치 포함), 내생 QoS, CMDP 비용
신호, 강력 베이스라인, 브랜치 통합 지점을 모두 검증합니다.

## 재현 — 학술지 확장 실험 (Phase C)

모든 실험은 단일 오케스트레이터로 실행하며, 산출물이 있으면 건너뛰므로
중단 후 재실행해도 안전합니다.

```bash
# 권장 실행 순서
python -m src.scripts.run_feasibility_probe          # 보정 증명서
python -m src.scripts.run_journal_experiments --stage c1_ppo
python -m src.scripts.run_journal_experiments --stage d_calib
python -m src.scripts.run_journal_experiments --stage c7_bo,c9_static
python -m src.scripts.run_journal_experiments --stage c2_lagrangian
python -m src.scripts.run_journal_experiments --stage c3_dsweep,c10_negctrl
python -m src.scripts.run_journal_experiments --stage c5_myopic,c6_algos,c8_mpc
```

결과는 `results/journal/*.json`, 모델은 `models/journal/`, 실행 이력은
`results/journal/MANIFEST.json`(git SHA 포함)에 기록됩니다.

## 재현 — 학술대회 논문 (보존)

```bash
python -m src.scripts.make_paper_figure   # 그림 2 재생성
python -m src.scripts.run_churn_sweep     # PPO m-sweep 재학습
python -m src.scripts.run_myopic_sweep    # Myopic-PPO m-sweep
python -m src.train.run_multi_seed        # m=1 PPO/SAC/TD3
```

## 디렉터리

```
src/env/network_slicing_env.py  환경 (외생/내생 QoS, CMDP cost, cohort)
src/train/                      학습·베이스라인 (PPO/SAC/TD3/Myopic,
                                PPO-Lagrangian, BO 오라클, Oracle MPC,
                                휴리스틱, 통계 유틸)
src/scripts/                    실행 진입점 (오케스트레이터·프로브·그림)
tests/                          회귀·통합 테스트
results/, models/               실험 산출물 (공개 저장소에서는 제외)
```

## 재현성 주의사항

- 시드는 학습(42, 123, 456, 789, 1011, 1213, 1415)·평가(1000–1019)·
  탐색(BO/그리드, 42+)이 서로 분리되어 있습니다. 모든 정책은 동일한 평가
  시드를 사용하므로 정책 간 비교는 paired design입니다.
- **비트단위 재현은 동일 플랫폼/BLAS에서만 보장**됩니다. OS·하드웨어가
  다르면 부동소수점 연산 순서 차이로 궤적이 달라질 수 있으나, 통계적
  결론(시드 평균·신뢰구간)은 영향을 받지 않습니다.

## 라이선스

MIT — `LICENSE` 참조.

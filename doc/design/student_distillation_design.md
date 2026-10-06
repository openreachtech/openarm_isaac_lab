# Depth Student 蒸留 — 設計

Teacher（特権情報つき PPO）から、Depth 画像を入力とする Student へ方策蒸留するための設計。

- 参考論文: [DextrAH-G](../papars/DextrAH-G_Pixels-to-Action_Dexterous_Arm-Hand_Grasping_with_Geometric_Fabrics.md) — Student アーキテクチャ（F.3）、損失（3.3）、Depth 拡張（F.1）
- 参考実装: `unitree_rl_lab` の `origin/feat/wall` — `Go2-v3-Level1`（teacher）/ `Go2-v3-Student-Level1`（student）
  - 注: ローカル `feat/wall` は 4 コミット古く Student タスクを含まない。`origin/feat/wall` を参照すること。

決定事項: **カメラは固定（三人称）**、**80×60 / 256 envs** から始める。

---

## 1. 出発点：Teacher の実測仕様

`logs/rsl_rl/arm_lift_cube/2026-10-04_19-40-23/model_1999.pt`（基部 `z = 0.15` の現行 Teacher）より:

```
actor: 36 → [256, 128, 64] → 8    (MLP, 非再帰, elu)
```

観測 36 次元の内訳（`lift_env_cfg.py` の `ObservationsCfg.PolicyCfg`）:

| 項目 | 次元 | Student での扱い |
|---|---:|---|
| `joint_pos` (7 arm + 2 finger) | 9 | そのまま |
| `joint_vel` | 9 | そのまま |
| `last_action` | 8 | そのまま |
| `target_object_position` (pose command) | 7 | そのまま |
| `object_position` | 3 | **削除 → Depth から推定** |

行動 8 = アーム 7 関節 + グリッパ二値 1。

**設計の核心**: 特権情報は `object_position` の 3 次元だけ。これは DextrAH-G の `x_obj` と完全に一致するため、論文の「補助ヘッドで `x̂_obj ∈ R³` を予測させる」構成がそのまま適用できる。

### 特権観測のノイズ

`object_position` には観測ノイズを載せる（`lift_env_cfg.py` の `OBJECT_POSITION_NOISE_STD = 0.02`）。

| 成分 | 設定 | 役割 |
|---|---|---|
| 毎ステップ | `GaussianNoiseCfg(std=0.02, operation="add")` | 瞬時のばらつき |
| エピソード毎 | `GaussianNoiseCfg(std=0.02, operation="abs")` | 系統的なオフセット |

DextrAH-G は特権 Teacher の物体姿勢観測に `sigma_xyz,uncorr = sigma_xyz,corr = 0.02 m` を加えている（§3.2 "Pose Noise" / 付録 E.4）。狙いは、数 cm の誤差しか持てない Depth Student に操縦されても破綻しない Teacher にすること。真値前提で学習した Teacher は、Student が位置を 3 cm 外した瞬間の挙動が未定義になる。

2 成分に分ける理由は、**エピソード内で一定のバイアスこそが Student の誤差の実態**だから。毎ステップのノイズだけでは時間平均で消えてしまい、系統誤差への耐性が付かない。`operation="abs"` はリセット時にバイアスを引き直す指定で、`"add"` にすると既存バイアスへ加算されエピソード跨ぎのランダムウォークになる。

ノイズが載るのは Teacher の訓練時（`Arm-Lift-Cube`、`enable_corruption = True`）だけ。蒸留側の teacher 群は `enable_corruption = False` なので `ObservationManager` が `noise` を `None` に落とし、**Student が模倣するラベルは綺麗なまま**になる。`-Play` も同様にノイズ無し。

---

## 2. 観測設計

rsl_rl の `StudentTeacher` は observation group を平坦なベクトルとして受け取る。unitree と同じく、画像も平坦化して連結し、モデル側で幅スライスして復元する。

```python
obs_groups = {"policy": ["policy"], "teacher": ["teacher"]}
```

| group | 内容 | 次元 |
|---|---|---:|
| `policy`（student が見る） | `[proprio 26 \| goal 7 \| depth 4800]` | 4833 |
| `teacher`（凍結 teacher + 補助ラベル） | 既存の 36 次元 | 36 |

- `proprio 26` = `joint_pos 9` + `joint_vel 9` + `last_action 8`
- `goal 7` = `target_object_position`
- `depth 4800` = 80×60 **1ch**、`TiledCameraCfg(data_types=["distance_to_image_plane"])`

補助ラベル `x_obj` は `teacher` group の `object_position` スライス（オフセット 18..21）から取得する。

RGB は使わない。DextrAH-G も raw depth 単体（`I ∈ [0.5,1.5]^{160×120}`）。RGB を足すと 4ch になり storage が 4 倍、レンダリングコストも増える上、sim2real ギャップは depth のほうが小さい。

### シーン幾何

| 要素 | 値（env 座標） |
|---|---|
| テーブル天面 | `z = 0`（`[0.5, 0, 0]` 配置） |
| Cube 初期位置 | `[0.4, 0, 0.055]`（静止時は `z = 0.021`） |
| Cube リセット範囲 | x `[0.3, 0.5]`、y `[-0.25, 0.25]` |
| 目標位置範囲（基部相対） | x `[0.2, 0.4]`、y `[-0.2, 0.2]`、z `[0.15, 0.4]` |
| env_spacing | 2.5 m |

### アーム基部とカメラ

**最終的な実機構成は Go2 四足の上にアームを載せ、カメラを頭部（＝基部のすぐ下）に置く形**を想定している。Go2 の体高 30cm では OpenArm の長さが足りないため、本番ではより長いアームを使う。OpenArm はその前段の試験。

その縮小版として、アーム基部を持ち上げカメラを基部直下に置く:

| | 値（env 座標） |
|---|---|
| アーム基部 | `(0, 0, 0.15)` — 元は `(0, 0, 0)` |
| カメラ | `(0, 0, 0.10)` — 基部の 5cm 下 |
| 光軸 | **+x 水平**（俯角 0°） |
| FOV | 87° × 71°（4:3、RealSense D435 相当） |
| clipping | `(0.1, 1.2)` |

カメラ姿勢は env cfg の定数として切り出す。Level2 で摂動を加えるため、どのみち変数で持つ必要がある。

### 画角のカバー範囲

前方距離 `d` における可視範囲:

```
垂直: z ∈ [0.10 − 0.695d,  0.10 + 0.695d]
水平: y ∈ ±0.927d
```

水平画角は `2·atan(20.955 / (2·11.3)) = 85.7°`、垂直はアスペクト比より `69.6°`。

| d | 垂直 | 水平 | 判定 |
|---|---|---|---|
| 0.3（Cube 手前端） | `−0.11 … 0.31` | `±0.278` | 要 y `±0.25` + Cube 半幅 0.021 = 0.271 → ✓（**余裕 7mm**） |
| 0.5（Cube 奥端） | `−0.25 … 0.45` | `±0.464` | ✓ |

テーブル面（z=0）が見え始めるのは `d = 0.141` から。Cube 範囲 `x = 0.3–0.5` は全域カバーする。

**水平（俯角 0°）にした理由**: 俯角 9° では持ち上げた Cube が world `z ≈ 0.22` で画角外に出るが、水平なら `z ≈ 0.31–0.46` まで追える。画面の使い方としても、下半分がテーブルと静止中の Cube、上半分が持ち上げ・運搬領域となり無駄がない。

目標位置は world `0.30–0.55`（基部比 `0.15–0.4`）なので運搬の後半は画角外になるが、その時点では把持済みで位置は FK で復元できる。

### 物体配置の上限（2026-10-05 実測）

Cube の配置範囲 `x ∈ [0.30, 0.50]`, `y ∈ ±0.25` は、**二つの独立した制約が箱の反対側の隅でほぼ同時に飽和**している。広げる余地はほとんどない。

| 制約 | 内容 | 効く隅 | 余裕 |
|---|---|---|---|
| リーチ | TCP の最大到達半径 **0.616 m**。基部は卓面の 0.13 m 上なので卓上での水平リーチは `√(0.616² − 0.13²) = 0.602 m`。上からの把持はさらにアームを畳むので実効値はこれより小さい | 遠隅 `(0.50, 0.25)` = 基部から水平 **0.559 m** | 約 7% |
| 画角 | 水平 FOV 85.7° の水平カメラなので距離 `x` で `|y| < 0.927x` | 近隅 `(0.30, 0.25)` = 0.271 必要 / 0.278 可視 | **7 mm** |

テーブルは制約にならない。`x ∈ [0.10, 0.80]`, `y ∈ [−0.60, 0.60]` で 3072 回落下させて脱落ゼロ。

**yaw は空いている**ので `±π` でフルランダム化した。Cube の一辺 0.042 m（面対角 0.059 m）に対しグリッパ開口は 0.088 m、yaw ランダムで 128 回落として 100% 平らに着地する。ただし Teacher の観測は位置のみで**姿勢を含まない**ため、yaw に合わせて向きを変えるのではなく、どの yaw でも通る把持を学ぶことになる。

x/y をこれ以上広げるにはハードウェア側の変更が要る（より長いアーム、より広い FOV、またはカメラに仰角）。仰角は上限 17°（Cube 範囲が `x = 0.30` から始まるため、それ以上傾けるとテーブル手前が切れる）。

### エピソード初期化のランダム化

`EventCfg`（`mode="reset"`）で毎エピソード振るもの:

| 対象 | 範囲 | 頻度 | 根拠 |
|---|---|---|---|
| Cube 位置 | `x ±0.1`, `y ±0.25` | 毎エピソード | 上記の通り飽和済み |
| Cube yaw | `±π` | 毎エピソード | 唯一空いていた軸 |
| ~~アーム関節~~ | **不採用** | — | 下記のアブレーション参照 |
| Cube サイズ | scale `0.72–0.88`（一辺 3.72–4.68 cm） | **env ごとに固定** | 下記 |

初期アーム姿勢のランダム化（`±0.1 rad`）は**実測の結果、不採用**とした。TCP が 3〜4 cm 動くだけで、しかも `joint_pos` で観測できるため安全だと考えていたが、アブレーションでは試した変更のうち**最も有害**だった。

| 変更 | final `lifting_object` | baseline 比 |
|---|---|---|
| baseline（2048 / 4096 envs） | 12.754 / 12.969 | 100% |
| Cube サイズ `±11%` | 12.756 | 100% |
| 観測ノイズ `σ=0.02` | 7.016 | 55% |
| Cube yaw `±π` | 0.992 | 8% |
| **初期アーム姿勢 `±0.1 rad`** | **0.232** | **2%** |
| 上記4つすべて | 0.121 | 1% |
| **初期アーム姿勢のみ除外** | **8.952** | **70%** |

最後の2行の差は `reset_robot_joints` の有無だけ。これを外すと `0.121 → 8.952` になる。

未検証の説明として、オフセットを手首を含む 7 関節すべてに入れているためグリッパの**向き**も変わり、Cube yaw と同じ理由（グリッパと Cube の固定された相対姿勢の破壊）で失敗している可能性がある。関節 1〜4 に限定すれば切り分けられる。初期姿勢の多様性が必要になったらそこから試す。

`±0.2 rad` 以上も物理的には問題ない`±0.2 rad` 以上も物理的には問題ない（TCP で 7〜8 cm）が、特権観測ノイズと yaw フルランダム化を同時に入れているため、まず `±0.1` で Teacher が学習できることを確認してから上げる。

### 遮蔽特性

カメラを基部の**下**に置くことで、接近フェーズの遮蔽が避けられる。基部からアームが前方へ伸び、グリッパは Cube の上から降りてくるため、カメラ視点ではグリッパが Cube の上側に見えるだけで、最終降下まで Cube を覆わない。

結果として遮蔽は「把持直前〜把持後」に限定され、`object_position ≈ FK(joint_pos)` で復元できる。これが §3「削除 2」で再帰層を外せる根拠を支えている。基部の**上**に置くと接近中に視線とアームが重なり、この根拠が崩れる。

---

## 3. モデル

DextrAH-G F.3 をベースに、本タスクに合わせて 2 点削っている（再帰層、状態側エンコーダ）。

```
 proprio(26)  goal(7)      depth (1×60×80)
     │          │                │
     │          │        conv 16/32/64 (k3,s1,p1) + pool ×3
     │          │                │
     │          │           flatten 4480
     │          │                │
     │          │          MLP 128→128 (relu)
     │          │                │
     └──────────┴────────┬──────128
                    concat 161
                         │
                 MLP 256→128→64 (elu)
                         │
               ┌─────────┴─────────┐
             â (8)            x̂_obj (3)
```

Conv スタック（k=3, s=1, p=1 / MaxPool k=2, s=2 / ReLU）:

| 層 | 出力 |
|---|---|
| input | 1×60×80 |
| conv1 (→16) + pool | 16×30×40 |
| conv2 (→32) + pool | 32×15×20 |
| conv3 (→64) + pool | 64×7×10 |
| flatten | 4480 |
| MLP(128,128) | 128 |

depth 側は論文の指定どおり残している。削減後の最大コストは `4480→128` の全結合（約 573k）で、さらに削るならここに pool を一段足すか global pooling にする。

### 削除 1: proprio / goal のエンコーダ

論文は `o_robot` と `x_goal` をそれぞれ MLP(512,256,128) で符号化するが、本タスクでは**生のまま trunk に入れる**。

合計 33 次元を 512 幅で符号化するのは過剰で、この 2 本だけで約 344k パラメータ（Teacher 全体より大きい）になる。Teacher 自身が 36 次元の生の状態を `[256,128,64]` で直接行動に写しており、低次元の固有感覚に深いエンコーダが不要なことを示している。

trunk を `[256,128,64]` としたのは Teacher と同容量だから。Student の仕事は「Teacher の関数 ∘ (depth → object_position)」なので、状態側は Teacher と同容量あれば足りる。

### 削除 2: 再帰層（GRU）

論文が GRU を採る理由は 2 つあるが、どちらも本タスクには当てはまらない。

1. 「Teacher が LSTM だから Student も state-based にした」(F.3) — 本件の Teacher は非再帰 MLP なので無関係。
2. 「遮蔽中に物体位置を推論するため」(F.3) — 本タスクでは遮蔽局面で `object_position` が proprio から復元できる。

| 局面 | Cube の可視性 | `object_position` の情報源 |
|---|---|---|
| 接近 | 見えている | 単一 Depth フレームで十分 |
| 把持後・運搬 | グリッパが遮蔽 | ≈ FK(`joint_pos`) — proprio から復元可能 |

把持後は Cube がハンドに固定されるため位置はほぼ順運動学で決まり、Student は `joint_pos` を持っている。記憶が要るのは「接近したがまだ掴んでいない」短い遷移だけ。

論文自身も代替を認めている:

> We opted for GRU, but we believe any form of a state-based model (RNN, LSTM, GRU) or a stateless model (MLP) with history as concatenated input should work.

非再帰にすることで実装が大きく軽くなる:

| | GRU 版 | 本設計 (MLP) |
|---|---|---|
| `is_recurrent` | `True` | `False` |
| hidden state の reset/detach/復元 | 必要 | 不要（no-op） |
| `gradient_length` | TBPTT 長 | 単なる勾配累積 |
| rsl_rl `StudentTeacher` | 再帰対応の上書きが必要 | ほぼ素で使える |

### 正規化

生の proprio/goal を trunk に直接入れるため、スケールが揃っている必要がある。`joint_pos_rel` / `joint_vel_rel` は相対値なので概ね問題ない。depth は clipping range で `[0,1]` に正規化してから conv に入れる。

rsl_rl の `student_obs_normalization` は平坦化された観測全体（depth ピクセルを含む）に running mean/std を走らせてしまうため、unitree と同じく `False` にしてモデル側で明示的に正規化する。

### 履歴

履歴が必要と判明した場合の拡張口は `ObservationTermCfg(history_length=N)`。ただし Depth を N フレーム積むと storage が N 倍になる（80×60 / 4 フレームで約 475 MB）。まず単一フレームで `x̂_obj` の誤差を見て、把持遷移で悪化する場合のみ検討する。

---

## 4. 損失

```
L = ||â − a_teacher||₂ + β · ||x̂_obj − x_obj||₂        (β = 0.1)
```

unitree の `Distillation.update()` が `L = L_bc + coef · L_recon` という同形なので流用できる。差し替えるのは教師信号の取得のみ:

| | unitree | 本設計 |
|---|---|---|
| `L_bc` | student action vs teacher action | 同じ |
| 補助項 | height-map 再構成 (`get_clean_extero`) | `object_position` 回帰 |
| 係数 | `reconstruction_loss_coef = 0.5` | `aux_pos_loss_coef = 0.1`（論文 β） |

---

## 5. 追加・変更するファイル

```
source/openarm/openarm/
├── assets/models/                              # 新規
│   ├── depth_student.py                        # DepthStudentPolicy (conv + MLP + 2 heads)
│   └── modules/
│       ├── student_teacher.py                  # DepthStudentTeacher(rsl_rl.StudentTeacher)
│       ├── distillation.py                     # Distillation + 補助位置損失
│       └── runners.py                          # クラス登録フック
└── tasks/manager_based/openarm_manipulation/unimanual/lift/
    ├── config/
    │   ├── __init__.py                         # gym.register を 2 件追加
    │   ├── student_env_cfg.py                  # 新規: Depth カメラ + obs group
    │   └── agents/rsl_rl_distillation_cfg.py   # 新規: runner/policy/algorithm cfg
    └── lift_env_cfg.py                         # 変更なし（teacher 用をそのまま使う）
```

`scripts/rsl_rl/train.py:196` は `DistillationRunner` を直接構築しているため、unitree の `_register_distillation_classes` 相当のフックを挟む必要がある（rsl_rl が `eval(class_name)` でクラスを解決するため、モジュールスコープに注入する方式）。

### タスク ID

`Arm-Lift-Cube` の命名に合わせる:

| 役割 | タスク ID |
|---|---|
| Teacher（既存） | `Arm-Lift-Cube` |
| Student Level1 | `Arm-Lift-Cube-Student-Level1` |
| Student Level2 | `Arm-Lift-Cube-Student-Level2` |

`rplay arm_lift_cube_student_phase1` で引けることを確認済み。

---

## 6. 主要な設定値

```python
num_steps_per_env   = 24      # teacher PPO と同じ
gradient_length     = 4       # 非再帰なので「実効ミニバッチ」の意味 (num_steps_per_env % gradient_length == 0 が必要)
num_learning_epochs = 2
learning_rate       = 5.0e-4  # unitree 由来。DextrAH-G には記載なし
max_grad_norm       = 1.0
aux_pos_loss_coef   = 0.1     # 論文 β
num_envs            = 256
```

storage 概算: `24 × 256 × 4833 × 4B ≈ 119 MB`（RTX 5080 16GB に対して十分。レンダリング分は別途）。

解像度を 160×120（論文値）に上げると storage は約 472 MB。まず 80×60 でパイプラインを通し、動いてから上げる。

### `gradient_length` の意味（再帰層を外したことによる変化）

`rsl_rl/storage/rollout_storage.py:154-159` の generator は **1 タイムステップ分（全 env バッチ）を時系列順に** yield する。`Distillation.update()` はそれを `gradient_length` 回ぶん **和で**累積してから 1 回 backward する。

```python
loss = loss + behavior_loss      # 平均ではなく和
if cnt % self.gradient_length == 0:
    loss.backward(); optimizer.step(); loss = 0
```

再帰モデルならこれは TBPTT だが、**非再帰では各ステップの forward が独立なので単なる勾配累積＝実効ミニバッチサイズ**になる。

| `gradient_length` | 実効バッチ | 1 iteration あたりの最適化ステップ |
|---:|---:|---:|
| 8 | 2048 | 6 |
| **4** | **1024** | **12** |

レンダリングがボトルネックなので、1 iteration で払う描画コスト（256 envs × 24 step）に対して更新回数を稼げる 4 を採る。256 envs 分の独立軌道が各バッチに入るため 1024 でも conv net の学習には十分。

注意点 2 つ:

- **損失は和で累積される**ため、`gradient_length` を変えると勾配の大きさも比例して変わり、本来は learning_rate と連動する。ただし `max_grad_norm = 1.0` でクリップされる領域なら実効ステップ幅は一定。最初の数十 iteration で grad norm を監視し、常時クリップされているか確認する。
- **シャッフルされない**。ミニバッチは連続する `gradient_length` ステップ × 全 env。env 間は独立なので実害は小さいが、大きくするほど時間的相関が強くなる。これも小さめの値を推す理由。

---

## 7. Depth 拡張（Level2）

論文 F.1 より。Level1 ではすべて無効、Level2 で投入する。

| 拡張 | パラメータ |
|---|---|
| ピクセル脱落（0 埋め） | `p_dropout = 0.003` |
| ランダム値置換 | `p_randu = 0.003` |
| 線状アーティファクト（配線模擬） | `p_stick = 0.0025`、長さ ≤18px、幅 3px |
| 相関 / 無相関ノイズ | 論文 [57] のモデル |
| カメラ位置の摂動 | 校正誤差への頑健性 |

論文の depth clipping は `[0.5, 1.5] m`。卓上 Lift ではカメラ〜テーブル距離が異なるため、実配置に合わせて再設定する。

---

## 8. 難易度レベル

一度に全部を課さず、段階的に難易度を上げる。

**この段階分けは DextrAH-G 由来ではない。** 論文は Teacher / Student の 2 段しか分けておらず、F.1 の Depth 拡張は蒸留中ずっと有効でランプ投入しない。段階分けは Miki et al. S3（unitree の Student-Phase1 / Phase2 が倣っている構成）から採っている。本設計は「DextrAH-G のアーキテクチャと損失」＋「Miki の段階的学習」の組み合わせ。

Miki の構成では、まず理想的な知覚のもとで写像を獲得させ、その後で情報を削っていく。各レベルは1つ下のチェックポイントから `--resume` で継続する。

| | Depth | カメラ位置 | ねらい |
|---|---|---|---|
| **Level1** | クリーン | 固定 | teacher の模倣と `x̂_obj` 推定の獲得 |
| Level2 | クリーン | 摂動あり | 実機カメラの校正誤差への頑健化 |
| Level3+ | §7 の拡張を投入 | 摂動あり | センサノイズへの頑健化 |

Level2 以降は未実装。番号は意図的に開いてあり、難易度の軸（物体形状、テーブル高さ、外乱など）を増やす余地を残している。

レベルを分ける理由は、Level1 が**設計の前提を検証する役割**も兼ねているため。理想的な知覚で `x̂_obj` が収束しないなら、それはノイズの問題ではなく単一フレーム・非再帰という設計判断の問題だと切り分けられる（§10）。

---

## 9. 実行方法

Student タスクはカメラを持つので **`--enable_cameras` が必須**。teacher のチェックポイントは別タスク名のログディレクトリにあるため `--teacher_task` で指定する（これがないと student 自身の空のログディレクトリを探して失敗する）。

```bash
# Level1: teacher から蒸留
python scripts/rsl_rl/train.py \
  --task Arm-Lift-Cube-Student-Level1 --teacher_task Arm-Lift-Cube \
  --headless --enable_cameras

# 再開（teacher ではなく student 自身の run を読む）
python scripts/rsl_rl/train.py \
  --task Arm-Lift-Cube-Student-Level1 --resume --load_run <run> \
  --headless --enable_cameras
```

### 検証済みの実測値

2026-10-04 時点、3 iteration の結合テストと 9 env のレンダリングで確認:

| 項目 | 実測 |
|---|---|
| `policy` 観測 | `(4833,)` = 9 + 9 + 8 + 7 + 4800（depth が末尾） |
| `teacher` 観測 | `(36,)`、`object_position` は index 18–21 |
| conv 出力 | `64×7×10 = 4480` → head 128 → trunk 入力 161 |
| Student パラメータ | 696,715（`cnn.head` が 590,080 で最大） |
| Teacher ロード | `strict=True` で全キー一致（素の `MLP([256,128,64], elu)`） |
| depth 値域 | `0.044 – 1.000`、全て有限 |
| far plane 占有率 | **58.7%** |
| Cube の可視性 | 全フレームで確認（約 8px 四方） |

far plane が 59% を占めるのは、水平カメラで静止時にテーブル上空に何も無いため。そこは持ち上げ後に Cube が通る領域なので恒久的な無駄ではないが、俯角をつけるかどうかの判断材料になる。

Cube が 80×60 で約 8px しかない点は、位置回帰の精度上限に効く可能性がある。`x̂_obj` の誤差が頭打ちになるようなら解像度を上げる。

---

## 10. 未解決・リスク

解決済み（2026-10-04）:

- ~~**Teacher の再学習**~~ 完了。基部 `z = 0.15` で 2000 iter 再学習し、`lifting_object 13.05` / `object_dropping 0.000` / `position_error 0.085`。基部 `z = 0` のとき（13.94 / 0.000 / 0.083）をやや下回るが、掴んで保持できている。run: `logs/rsl_rl/arm_lift_cube/2026-10-04_19-40-23`。
- ~~**基部を上げた状態での到達可能性**~~ 上記の再学習が成立したことで確認済み。なお基部〜Cube の距離自体はほとんど変わらない（最遠隅で 0.562 → 0.567 m）。変わるのは向きで、`dz` が `+0.055` から `−0.095` になる。
- ~~**カメラ画角**~~ 描画で確認済み（§9）。Cube は全フレームに写る。

残り:

- **単一フレームで `object_position` を十分な精度で推定できるか**が最大の未検証点。本設計が再帰を落とした根拠そのものなので、Level1 で `x̂_obj` の誤差を局面別（接近 / 把持遷移 / 運搬）に見る。把持遷移で明確に悪化するなら `history_length` を検討する。
- **レンダリングのスループット未計測**。256 envs での TiledCamera のコストが訓練時間を支配する可能性がある。Level1 の最初の数百イテレーションで実測する。
- **特権観測ノイズ / Cube yaw / Cube サイズを入れた後の Teacher 再学習が未実施**（sandbox の try8 で学習成立は確認済み、本番 run は未実施）。いずれも Teacher の訓練分布を変えるため、現行の Teacher (`2026-10-04_19-40-23`) と Student (`2026-10-04_20-42-53`) はいずれもノイズ無しで学習されたもの。Teacher → Student の順で再学習が要る。
- **ゴール追従精度**（`object_goal_tracking_fine_grained` が Teacher 比 69%）。掴む動作自体は Teacher 比 98.8% で、落ちているのは掴んだ後の寄せの精度のみ。CNN の空間量子化（最終特徴マップ 1 セル = 入力 8px = 物体位置で約 6 cm、報酬カーネル幅 5 cm と同オーダー）が有力な容疑だが未検証。当面は許容し、表面化した時点で再検討する。論文との差分は解像度（160×120 対 80×60）、カメラ FOV（D415 の 65° 対 86°）、時間方向の積分（GRU / フレームスタックの有無）の 3 点。
- **Teacher が非再帰（MLP）** である点。論文は state-based teacher のほうが成功率が高いと述べているため、Teacher 自体の性能上限は論文より低い可能性がある。ただし Student は Teacher を模倣するだけなので、蒸留の成否とは独立。

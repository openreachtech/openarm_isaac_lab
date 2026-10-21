# Arm-Place-Sphere — 設計

球を掴んで、低い壁つきトレイ（ビン）に入れて離すタスク。ManiSkill の `PlaceSphere-v1` を OpenArm（単腕）に移植する。

- 移植元: `ManiSkill/mani_skill/envs/tasks/tabletop/place_sphere.py`（ManiSkill3 側も中身は同一）
- 先行タスク: `Arm-Lift` — ロボット、基部高さ、腕アクション、連続グリッパ、RMA の観測構成、初期姿勢カリキュラムを継承する
- タスク ID: **`Arm-Place-Sphere`**（teacher）/ **`Arm-Place-Sphere-Student`**（深度カメラ、§13）

---

## 1. ManiSkill 版の要点

| 項目 | ManiSkill `PlaceSphere-v1` |
|---|---|
| 物体 | 球 r = 2 cm（dynamic） |
| ビン | kinematic。底板 5 cm 四方 × 厚さ 5 mm、壁 4 枚は厚さ 5 mm × 高さ 1 cm。内寸 4 cm 四方で、球がぴったり収まる |
| 配置 | 球 x∈[-0.10,-0.05]、ビン x∈[0,0.10]、どちらも y∈[-0.1,0.1]（テーブル中心基準。ロボットは x=-0.615 にいる） |
| 成功 | 球とビンの xy 距離 ≤ 5 mm、かつ z が底板の上 ±5 mm、かつ球が静止、かつ把持していない |
| 報酬 | 段階式で最大 13。到達 0–2 → 把持して運ぶ 4–5 → ビン上で離して静止 6–12 → 成功 13 |
| 長さ | 50 step（20 Hz で 2.5 s） |

---

## 2. 方針

**Arm-Lift を継承し、ゴールを「空中の点」から「ビンの中の点」に差し替える。**

`object_pose` コマンドは、観測（`target_object_position`、`object_to_goal`）、報酬（goal tracking）、カリキュラム（`arm_init_level`）の 3 か所から参照されている。このコマンドを**ビンの位置から目標点を計算するコマンド**に置き換えれば、既存の項目のほとんどがそのまま動く。新しく作るのは次の 4 つに限られる。

1. ビン本体
2. ビンの配置
3. 「ビンに入った・離した」に関する報酬
4. カリキュラムの昇格条件

物体は球 1 種類にする。ただし catalog の仕組み（`object_catalog()` のバッファ、`lift_threshold`、質量・摩擦のランダム化、特権観測）は**エントリ 1 つの catalog としてそのまま使う**。こうすると特権ベクトルのレイアウト（`PRIVILEGED_CATEGORY_SLICE`）が変わらないので、`RmaActorCritic` と runner の設定をそのまま流用できる。

---

## 3. シーン

### 球

- `SphereCfg(radius=0.02)`。catalog の `_make_sphere` で半径を固定したものを使う。
- 密度と摩擦は Arm-Lift と同じく、エピソードごとにランダム化する（`DENSITY_RANGE`、`FRICTION_RANGE`）。転がりやすさが変わるので、このタスクでは Arm-Lift より効果が出る見込みがある。

### ビン（トレイ）

- **寸法は ManiSkill と同じ**（内寸 4 cm、壁の高さ 1 cm、壁と底の厚さ 5 mm）。
- 内寸を `BIN_INNER_HALF` として定数にしておく。精度が足りない場合は、まず 0.025 m（隙間 5 mm）に広げる。
- Isaac Lab のスポナーは 1 プリムに 1 形状しか作れない。そのため、**底板と壁 4 枚を子 Cube に持つ USD を pxr で生成**し、`UsdFileCfg` で 1 つの kinematic `RigidObject` として読み込む。
  - 生成した USD は寸法をファイル名に含めてキャッシュする。
  - 剛体のプロパティは `RigidBodyPropertiesCfg(kinematic_enabled=True)` にする。
- 指と壁の干渉について：
  - 球がビンに入っているとき、球の中心は卓面から 2.5 cm、壁の上端は 1.5 cm の高さにある。
  - 指は赤道（2.5 cm）付近で球を挟むので、壁の上に約 1 cm の余裕がある。
  - ただし腕の姿勢によっては指先がもっと下がる。実装後に確認が必要。

### 配置（OpenArm の作業域に合わせて変更）

ManiSkill の配置をそのまま使うと、球はロボット基部から 0.52–0.57 m、ビンは 0.62–0.72 m の距離になる。Arm-Lift で測定した把持失敗率は、0.45–0.50 m で 36.7%、0.50 m 以上で 75.6% だったので、**ManiSkill の配置ではまったく届かない**。そこで、配置は OpenArm の作業域に合わせて作り直す。

- **箱**：球もビンも、Arm-Lift と同じ箱 x∈[0.30, 0.42]、y∈[-0.18, 0.18]（ロボット基部の座標系）からサンプリングする。この範囲は、届く距離とカメラの視野（FOV）の両方ですでに検証済み。
- **距離**：球とビンの中心間距離は d∈[0.08, 0.20] m にする。
  - 0.08 m 以上なら、球の半径 2 cm とビンの外寸の半対角 3.5 cm を足しても重ならない。
  - ManiSkill の実効的な距離範囲 0.05–0.28 m とほぼ同じスケールになる。
- **サンプリング手順**：
  1. 球の位置を決める。
  2. 方向 θ をランダムに選び、距離 d だけ離れた位置にビンを置く。
  3. ビンが箱からはみ出た場合は引き直す（最大 10 回、ベクトル化して実行）。
- ビンの yaw は 0 で固定する。ManiSkill も固定している。
- 実装場所は新しいイベント `reset_bin_near_object`。既存の `reset_object_stable_pose` の後に実行し、球の新しい位置を読んでからビンを置く。
- 10 回とも箱からはみ出た環境では、y 方向の反対側にビンを置く。箱は y 方向に 0.36 m あるので、必ず収まる。

---

## 4. コマンド：`BinTargetCommand`

- 出力は、ロボット基部座標系でのビン目標姿勢（7 次元）。位置は `ビン中心 + (0, 0, 底板の半分の厚さ + 球の半径)` で、球がビンに収まったときの中心位置になる。
- `_update_command` で毎ステップビンの姿勢から計算し直し、再サンプリングはしない。
- Isaac Lab の `_reset_idx` は、イベント（reset）→ コマンドのリセットの順に実行される。そのため、ビンの配置とコマンドが食い違うことはない。
- 既存の `generated_commands`、`object_to_goal`、`arm_init_level` が、このコマンドを通して無変更で「ビンの中」を指すようになる。

---

## 5. 観測

**policy 群**：Arm-Lift と同じで、次元数 42 も変わらない。

- `target_object_position` の意味が「ビンの目標点」に変わる。
- `object_to_goal` の意味が「球から目標点への相対ベクトル」に変わる。
- ビンの位置には、ひとまずノイズを入れない。
  - 実機ではビンもカメラで推定することになるので、そのときは `object_position` と同じ σ=0.02 のノイズとエピソード固定バイアスを掛ける。
  - ただし最初からノイズを入れると、精度 5 mm のタスクが立ち上がらない恐れがある。そこで、学習が立ち上がってから追加する段階的な方針にする。

**privileged 群**：Arm-Lift と同じ 18 次元。

- 球なので `object_extent` と `object_category` は定数になる。レイアウトを保つために残す。
- `object_rot_6d` は球では意味がない。ただし、ずらすとスライス位置が変わってしまうので残す。
- 転がりを見るなら `object_lin_vel` が欲しくなるかもしれない。入れずに学習しても、静止の報酬は問題なく立ち上がった（§12）。

---

## 6. 把持判定

成功条件と報酬で「掴んでいない」かどうかを判定する必要がある。ManiSkill の `is_grasping` は、左右の指それぞれと物体との接触力が 0.5 N 以上であることを条件にしている。

- 物体だけに絞った接触センサを、**指 1 本につき 1 つ**（`left_finger_object_contact`、`right_finger_object_contact`）追加する。
  - 設定は `ContactSensorCfg(prim_path=指, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"])`。
  - 1 つのセンサで両方の指を拾うと、PhysX が起動時に `Filter pattern ... expected 128, found 64` を出して失敗する。PhysX はセンサの body 1 つにつき、フィルタが 1 つ一致することを要求するため。
- 既存の `finger_contact` は使わない。テーブルとの接触も含んだ合力なので、物体との接触だけを判定できないため。
- `is_grasped` = 両指の `force_matrix_w` のノルムがどちらも 0.5 N 以上。
- ManiSkill にある角度条件（85°）は最初は省く。

---

## 7. 報酬

Isaac Lab は報酬を項の**加算**で計算するので、ManiSkill の「段階ごとに値を上書きする」方式は使えない。代わりに、**次の段階に進むと 1 ステップあたりの報酬の合計が必ず増える**ように重みを決める。

| 項 | 関数 | 重み | ゲート |
|---|---|---|---|
| `reaching_object` | `object_ee_distance(std=0.1)` | 1.1 | — |
| `lifting_object` | `object_is_lifted_relative(margin=0.02)` | 15 | — |
| `place_tracking` | 目標点への距離に tanh（std=0.3） | 16 | 持ち上げている **または** ビンの真上にある |
| `place_tracking_fine` | 同上（std=0.05） | 5 | 同上 |
| `object_in_bin` | 球がビンに入っている（xy と z の成功条件） | **20** | — |
| `released_in_bin` | ビンに入っていて、把持しておらず、グリッパの開度（0–1） | 10 | — |
| `static_in_bin` | ビンに入っていて、`1 − tanh(10·v + ω)` | 5 | — |
| `place_success` | 成功条件 | 10 | — |
| `action_rate` / `joint_vel` | 既存のまま（`SMOOTHNESS_PENALTY_ONSET` のカリキュラムも含む） | -1e-4 → -1e-1 | — |

### 段階ごとの 1 ステップあたりの最大値（単調に増えることの確認）

| 段階 | 主な項 | 合計 |
|---|---|---|
| 卓上で球に到達 | reach | ≈ 1.1 |
| 持ち上げて、ビンの 3 cm 上で保持 | reach + lift + tracking 14.4 + fine 2.3 | ≈ 32.8 |
| ビンの中で、まだ握っている | reach + tracking 21 + in_bin 20（lift は 0 になる） | ≈ 42 |
| 離して静止し、成功 | + released 10 + static 5 + success 10 | ≈ 67 |

### 設計上の判断

**tracking のゲートを「持ち上げている または ビンの真上」にする。** Arm-Lift のゲートは「接地高さ + 2 cm より上」だが、ビンに入った球の中心は接地高さより 0.5 cm 高いだけなので、このゲートを通らない。そのままでは、球をビンに下ろした瞬間に tracking の報酬 21 が消え、方策が下ろさなくなる。「ビンの真上」は xy 距離がビンの外寸の半分（2.5 cm）以内とする。卓上の球はビンの壁と重なるので、この領域には入れない。

**`object_in_bin` を lift より大きくする。** ビンに下ろすと lift の報酬 15 を失う。in_bin の 20 がそれを上回るので、下ろすほうが得になる。

**離すことへの誘因は加算で作る。** ManiSkill は手を離した瞬間の `ungrasp=16` で大きく跳ねるが、ここでは released + static + success の合計 +25 で同じ効果を作る。報酬の段差が 1 ステップに集中しない分、rsl_rl の価値関数には優しいはず。

**静止判定では、鉛直軸まわりの回転を無視する。** ManiSkill の静止条件は角速度のノルムが 0.5 rad/s 以下だが、ここでは水平成分（転がり）だけを見る（`horizontal_ang_vel`）。PhysX では球と床の接触が 1 点なので、滑り摩擦は法線まわりにトルクを生まない。ねじり摩擦も、`torsional_patch_radius` を設定しない限り働かない。そのため、離すときに付いたその場での回転が永久に止まらない。最初は ManiSkill どおりの判定で学習・評価したが、失敗の 91% がこれだった（§12）。その場で回っていても位置はずれないので、置けていると判定してよい。同じ理由で、`static_in_bin` の ω も水平成分だけにした。

**成功時に終了しない。** 終了すると、それ以降の正の報酬を失うので、成功が罰になってしまう。ManiSkill の PPO は成功時に部分リセットするが、Arm-Lift もゴール到達では終了しない。それに合わせる。

---

## 8. 終了条件

- `time_out`
- `object_dropping`（既存。z < -0.05）

---

## 9. カリキュラム

`arm_init_level` はそのまま使えない。次の 2 つの問題がある。

- 「保持している」の判定が `z > 0.04`（絶対高さ）になっている。ビンの中の球は中心が 0.025 なので、成功した環境でも「落とした」と判定され、**降格してしまう**。
- 昇格条件が「ゴールから 10 cm 以内」なので、ビンの横に置いただけでも昇格してしまう。

そこで `arm_init_level_place` を作る。

| 判定 | 条件 |
|---|---|
| 昇格 | エピソード終了時に成功している |
| 据え置き | 持ち上げているか、ビンに入っている |
| 降格 | 上記のどれでもない（卓上に残した、落とした） |

レベルの保持や `reset_arm_joints_by_level` は既存のものを使う。

---

## 10. その他の設定

| 項目 | 値 | 理由 |
|---|---|---|
| `episode_length_s` | 6.0 | Arm-Lift の 5 s に、下ろして離す動作の分を足した |
| `num_envs` | 2048（まずは Arm-Lift と同じ） | 球 1 種類なら `replicate_physics=True` で 4096 にできる可能性がある。ただし `randomize_object_physics` は False の前提で書かれているので、要検証 |
| runner | `ArmLiftRmaPPORunnerCfg` をそのまま使う | 観測の次元もレイアウトも同じ |
| PLAY | Arm-Lift の PLAY と同じ | ノイズなし、レベル 10 で固定 |

---

## 11. ファイル

| ファイル | 内容 |
|---|---|
| `unimanual/place/__init__.py`, `place/config/__init__.py` | 新規。`Arm-Place-Sphere` を登録（PLAY は `play_env_cfg_entry_point`） |
| `place/config/arm_place_sphere_env_cfg.py` | 新規。`ArmPlaceSphereEnvCfg(ArmLiftEnvCfg)` と `_PLAY` |
| `place/config/arm_place_sphere_student_env_cfg.py` | 新規。`ArmPlaceSphereStudentEnvCfg` と `_PLAY`（§13） |
| `place/bin_asset.py` | 新規。トレイの USD を pxr で生成し、キャッシュする |
| `place/mdp/__init__.py` | 新規。lift の mdp を再エクスポートし、以下を追加 |
| `place/mdp/commands.py` | 新規。`BinTargetCommand(+Cfg)` |
| `place/mdp/events.py` | 新規。`reset_bin_near_object` |
| `place/mdp/rewards.py` | 新規。判定（`place_target_w`、`object_in_bin`、`object_over_bin`、`is_grasped`、`object_is_static`、`place_success`）と報酬項 |
| `place/mdp/curriculums.py` | 新規。`arm_init_level_place`、および成功率をログに出す `episode_place_success` |
| `lift/object_catalog.py` | 追加。`build_sphere_catalog(radius)` と、名前で catalog を選ぶ `build_object_catalog(name)` |
| `lift/mdp/events.py` | 変更。`object_catalog()` が `env.cfg.object_catalog`（名前の文字列）を読むようにした。設定がなければ `"primitives"` になるので、Arm-Lift の挙動は変わらない |
| `scripts/tools/eval_arm_place.py` | 新規。失敗を種類ごとに分類し、配置・方向・質量・摩擦の区間ごとに成功率を集計する。student にも対応し、`--clean_obs` で teacher の観測ノイズを切れる |
| `scripts/rsl_rl/play.py` | 変更。エピソードが終わった環境について `policy_nn.reset(dones)` を呼ぶ（§13） |

---

## 12. 測定結果（2026-10-21、`model_1999.pt`、seed 42、1 本のみ）

学習：2048 環境、2000 iteration、約 20 分。評価：`scripts/tools/eval_arm_place.py`、6000 エピソード、arm_init_level 10。

### 学習の推移（`Curriculum/place_success`）

| iteration | 300 | 400 | 600 | 1300 | 1900 |
|---|---|---|---|---|---|
| 成功率 | 5.6% | 47% | 82% | 69% | 85% |
| arm_init_level | 0.2 | 1.5 | 8.7 | 9.8 | 9.97 |

- 1300 での落ち込みは、`SMOOTHNESS_PENALTY_ONSET`（iteration 1250）の直後。1400 で回復した。
- 行動ノイズの標準偏差 `mean_noise_std` は、1200 で 3.2 まで膨らみ、ペナルティの切り替え後に 1.2 まで戻った。
- この列は ManiSkill どおりの静止判定で数えた値。

### 静止判定による違い

| 静止判定 | 成功率 | 失敗の内訳 |
|---|---|---|
| ManiSkill どおり（ω のノルム） | 88.5% | ビンの中で回転したまま 10.5%、落下 0.9% |
| 水平成分だけ（採用） | **98.7%** | 落下 1.13%、縁に乗る 0.08%、その他 0.07% |

「ビンの中で回転したまま」に分類されたエピソードの、終了時の状態：
- 線速度は 0、指の接触は 0 N、グリッパは全開。
- 角速度の中央値は 1.1 rad/s（p90 で 3.6 rad/s）で、すべて z 軸まわりだった。
- 質量に比例して増えた。15–50 g で 5.6%、120–170 g で 15.7%。

成功までの時間は、中央値 0.96 s、p90 1.32 s、p99 1.86 s。

### 残りの失敗は落下で、遠くへ運ぶときに起きる

| 条件 | 落下率 |
|---|---|
| ビンが基部から 0.42–0.46 m | 3.0%（他の区間は約 0.9%） |
| ビンの y が +0.09〜+0.19 | 2.1% |
| 球とビンの距離が 0.17 m 以上 | 2.2%（0.08–0.11 m では 0.7%） |
| 球からビンへの方向が「奥」または「左（+y）」 | 1.6–1.8%（「手前」「右」は 0.6–0.7%） |

質量と摩擦では差が出なかった。運んでいる途中で落ちたのか、ビンに入った後で弾かれたのかは、まだ分けて測っていない。

---

## 13. Student（`Arm-Place-Sphere-Student`）

構成は Arm-Lift-Student とまったく同じ（`doc/design/arm_lift_rma_design.md` §4、§11）。
- place の teacher は Arm-Lift の観測クラスをそのまま継承しているので、次のものを無変更で流用した。
  - `ArmLiftStudentObservationsCfg`（policy / state / privileged / aux の 4 つの観測グループ）
  - 深度カメラ
  - `ArmLiftStudentDistillationRunnerCfg`
- 新しく書いたのは、env 設定とタスク登録だけ。

```bash
python scripts/rsl_rl/train.py --task Arm-Place-Sphere-Student --teacher_task Arm-Place-Sphere \
    --load_run 2026-10-21_14-20-38 --checkpoint model_1999.pt --enable_cameras --headless
```

### 測定結果（2026-10-21、`arm_place_sphere_student/2026-10-21_16-45-49/model_1999.pt`）

128 環境、2000 iteration、約 45 分。

| iteration | 0 | 400 | 1200 | 1900 |
|---|---|---|---|---|
| behavior loss | 1.650 | 0.037 | 0.020 | 0.014 |
| latent loss | 0.180 | 0.022 | 0.017 | 0.013 |

400 iteration でほぼ収束した。Arm-Lift-Student の最終値（behavior 0.104、latent 0.030）より低い。

| | エピソード数 | 成功率 | 落下 |
|---|---|---|---|
| teacher（ノイズあり、学習時と同じ条件） | 6000 | 98.72% | 1.13% |
| teacher（`--clean_obs`） | 6000 | 98.98% | 0.88% |
| **student** | 3072 | **99.09%** | 0.72% |

- 条件を揃えた比較（teacher もノイズなし）では、teacher と student の差は誤差の範囲（±0.2 ポイント程度）に収まった。蒸留による性能の低下は見られない。
- Arm-Lift-Student は teacher より −3.0 ポイントだった。物体が球 1 種類なので、latent が表すべきものが質量と摩擦くらいしかないことが効いていると考えられる。
- 残りの失敗の傾向も teacher と同じ。ビンの y が +0.09 以上のときに 97.4%（teacher は 98.0%）。

### 注意

- **知覚は理想的な設定。** `state` には物体の位置とビンの目標位置が正確な値で入っている。student が深度画像から推定しているのは、latent だけ。保存した深度画像（`depth_capture/`）では、ビンはほとんど見えない（壁の高さ 1.5 cm、80×60 ピクセル）。位置にノイズを入れる段階に進むなら、ここが最初の壁になる。
- **hidden state のリセット。** student は GRU で、学習中は環境がリセットされるたびにその環境の hidden state をクリアしている。一方、`play.py` と `eval_arm_lift.py` はクリアしていなかった。今回、`play.py` と `eval_arm_place.py` にはリセットを入れた。`eval_arm_lift.py` は未修正のままなので、`arm_lift_rma_design.md` §11 の Arm-Lift-Student の 85.2% は、前のエピソードの記憶を引き継いだ状態で測った値の可能性がある（未検証）。

---

## 14. リスク（実装前に挙げたもの）と、その結果

1. **押し込みで成功してしまう抜け道**：壁の上端（1.5 cm）は球の半径（2 cm）より低いので、掴まずに転がして押し込める可能性がある。→ **起きなかった。** 6000 エピソード中、一度も持ち上げずに成功したのは 0 件。
2. **指と壁の干渉**：離す瞬間に指が壁に当たり、球が弾かれる可能性がある。PLAY で目視確認し、必要なら `BIN_INNER_HALF` を 0.025 m にする。
3. **精度 5 mm と観測ノイズ**：policy 群の `object_position` には σ=2 cm のノイズとバイアスがかかっている。ただし `tcp_to_object` はノイズのない値なので、把持した後の位置合わせは TCP 基準でできる見込み。ビン位置にノイズを入れるのは後回しにする。
4. **球の転がり**：アプローチ中に指が触れると、球が逃げる。Arm-Lift で横倒しの円柱が苦戦した問題と同じ種類。
5. **報酬の段差**：上の表の単調性は 1 ステップあたりの最大値で確認しただけ。→ **問題なかった。** `object_in_bin` は 300〜600 iteration で立ち上がった。

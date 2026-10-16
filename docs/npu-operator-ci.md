# ARM NPU 算子 CI

## 触发与变更判断

工作流为 `.github/workflows/npu-arm-tests.yml`，显示名称为 `NPU ARM Operator Tests`。
允许仓库为 `lightseekorg/tokenspeed` 或仓库变量 `TOKENSPEED_NPU_CI_REPOSITORY` 指定的完整 `owner/repository`；fork 需要显式设置该变量才能参与运行。
触发包括 main push、以 main 为目标的 `pull_request` 和 `workflow_dispatch`。
`select` 可在允许仓库运行；硬件 job 另需仓库变量 `TOKENSPEED_NPU_ARM_CI_ENABLED` 显式设置为 `true`。
该变量缺失或未启用时，硬件 job 跳过，不进入 Runner 队列；筛选命中不代表算子已经运行。
NPU 白名单使用独立变量 `TOKENSPEED_NPU_CI_REPOSITORY`；无需设置 GPU 工作流使用的共享仓库变量，避免因部署 NPU 而同时启用其他厂商调度。
PR 事件支持 `opened`、`synchronize`、`reopened`、`ready_for_review`、`labeled`；仅 `high priority` 的加标签事件参与运行，草稿 PR 不启动 NPU job。
`high priority` 不改变精度先于性能的顺序。独立的 `pr-test-npu-arm` 并发组按 PR 或 ref 分组，自动事件取消同组旧运行；手动事件不主动取消。
无关标签事件使用独立组，避免取消正在执行的有效测试。不使用 `pull_request_target` 执行测试代码。

GitHub 托管机器上的 `select` job 获取完整 Git 历史，并运行 `ci/npu_ops/changes.py`：

- push 比较 `before` 与 `after` 两端；首次 push 的全零 `before` 保守执行全量。当前 main 限定下，这一语义仅用于新建 main 或检查器独立调用。
- PR 验证 checkout 为 `GITHUB_SHA` 的真实 merge commit，父提交依次匹配事件中的 base SHA 和 head SHA；变更范围使用 `merge-base(base, head)..head`，避免把 base 独有变更算进 PR。测试执行 merge SHA，不能用 head SHA 替代。
- 手动运行所选 branch/tag 的 `GITHUB_SHA`，无条件执行全量。
- SHA、ref、Git 对象或比较失败使检查失败，不会作为无关变更跳过。标签 push 和分支删除不启动算子测试。

路径规则集中在 `ci/npu_ops/manifest.json`，覆盖 NPU 包、实际使用的共享 `tokenspeed-kernel/python/`、依赖配置、NPU CI 脚本及自身工作流。
`.md`、`.rst` 文档变更排除。区分大小写匹配，检查新增、修改、删除及重命名前后的路径，不限制文件数量。
命中任一相关路径即运行所有登记算子；无关变更输出 `should_run=false` 和 `reason=no-watched-changes`。
工作流不使用顶层 paths 过滤，以便无关变更也能留下可读的筛选结果。
没有修改 NVIDIA/AMD 的工作流、公共 path filter、任务 pipeline、取消或 retry 控制器。
NPU 实现及其回归测试位于 `ci/npu_ops/`，新增算子时无需修改 GPU 的 `test/**` 清单。

## 文件职责与调用关系

- `ci/npu_ops/changes.py`：校验清单、事件身份并计算相关变更。
- `ci/npu_ops/manifest.json`：登记监听路径、算子顺序、脚本路径和预期数量。
- `ci/npu_ops/expected-plans.json`：记录精度用例、性能配置、采样参数及关联哈希。
- `ci/npu_ops/run_step.sh`：checkout 后的 ARM/Linux 校验、主机锁、环境和缓存准备。
- `ci/npu_ops/run_phase.py`：按顺序执行一个阶段，管理子进程和超时，保存证据并返回退出码。
- `ci/npu_ops/reporting.py`：校验结构化日志，生成 JSON 报告和中文 Markdown 摘要。
- `ci/npu_ops/<operator>/accuracy.py` 与 `performance.py`：算子的真实精度与性能测试。
- `ci/npu_ops/tests/`：不需要 NPU 的真实 Git、子进程及报告回归测试。

NPU job 仅编写两个 step：全部算子精度测试、全部算子性能测试。
两个 step 共享 YAML anchor 中的 bootstrap；精度 step 在获取脚本前完成平台与事件检查，创建独立 run/attempt 目录并获取 SHA，随后才能调用仓内入口。
性能 step 仅在精度成功后复用该目录；两次均验证 `uname -m` 为 `aarch64` 或 `arm64`、系统为 Linux、SHA 一致且 checkout 没有修改或未跟踪文件。
clone 只使用已校验格式和仓库白名单的 `GITHUB_REPOSITORY`，从 `https://github.com/<owner>/<repository>.git` 获取公开源码，不转发访问令牌。实际解释器路径固定，测试脚本通过同一解释器的 `-B -u` 执行，checkout 中两个 kernel 包目录位于 `PYTHONPATH` 前部。

| 算子 | 精度用例数 | 性能配置数 | 性能样本数 |
| --- | --- | --- | --- |
| scatter | 5 | 6 | 30 |
| virtual_slots_to_local | 48 | 8 | 40 |

合计 53 个精度用例、14 组性能配置、70 个样本。每组预热 10 次，采集 5 个样本，每个样本测量 50 次执行。
中位耗时是 5 个样本各自单次平均耗时的中位数，单位毫秒。
当前策略 `record_only` 只校验执行与采样完整有效；`performance_verdict=not_evaluated`、`threshold_ms=null`。
没有性能达标阈值，`completed` 不代表性能达标。

## 失败、日志与资源清理

任一用例、配置、脚本、checkout、环境或必要证据保存失败，立即返回非零，停止后续测试。
已启动阶段中的后续算子记为 `not_run`；精度 step 失败后性能 step 跳过。
失败后只继续必要的证据保存和资源清理，报告或 Actions 摘要保存失败也使 step 失败。
每个脚本超时 600 秒，整个 job 超时 60 分钟；这些不是性能阈值。

stdout/stderr 合并后实时写入日志并刷新到控制台，Python 使用无缓冲输出。
每个算子打印开始、结束、耗时和 exit；运行超过 30 秒时持续打印 heartbeat。
原始异常保留，`::error::` annotation 对 `%`、CR、LF 转义；失败原因和中文摘要在同一 step 内生成。
checkout/环境阶段失败由 bootstrap 写入带环节名的失败摘要；执行器异常还会显示 traceback。

测试子进程使用独立进程组；完成、非零退出、超时或 SIGINT/SIGTERM 都尝试终止该组内的子进程。
进程不得通过新 session 脱离监督。强制 SIGKILL 或主机掉线可能留下部分日志，无法保证最终摘要。
主机锁覆盖每个阶段的环境与测试过程；清理完子进程后随执行器退出释放。

## Runner 部署契约与报告位置

`runs-on` 仅使用专用标签 `tokenspeed-npu-arm64`，不附加默认 `self-hosted` 标签。
默认不启用硬件 job；此文描述必须满足的部署要求，不表示上游已经拥有 Runner、NPU 或配套环境。
部署方需在仓库首页依次打开 **Settings → Secrets and variables → Actions → Variables**，设置以下仓库变量：

| 变量 | 用途 |
| --- | --- |
| `TOKENSPEED_NPU_CI_REPOSITORY` | fork 设置为自身完整 `owner/repository`；官方仓库 `lightseekorg/tokenspeed` 已在白名单内，无需设置 |
| `TOKENSPEED_NPU_ARM_RUNNER_NAME` | 必须与 GitHub 中注册的 Runner 名称及运行时 `RUNNER_NAME` 一致，没有默认名称 |
| `TOKENSPEED_NPU_ARM_CI_ENABLED` | 确认全部部署要求后才设置为 `true`；保持未配置即可禁用硬件 job |

部署方必须把专用标签配置到 Linux ARM64 NPU Runner，确认服务在线，服务账号必须为 `ci-npu`。
启用后若 Runner 名称变量缺失或不匹配，step 在 checkout 前明确失败；若没有在线且带标签的 Runner，job 会等待调度，无法执行运行时检查。
环境入口为 `/srv/tokenspeed-npu/cann9-torch29-py312-root/env.sh`，解释器为该目录下 `venv/bin/python`。
必须部署可用的 CANN、torch、torch_npu 和 Triton Ascend 依赖，以及可供该账号使用的 NPU 设备；仅安装 Runner 不足以运行测试。
主机锁为 `/srv/npu-ci/locks/host-all.lock`，账号需有读写权限；锁被占用时直接失败，不启动测试。
必须预先创建 `/srv/npu-ci/jobs/slot0` 并允许 `ci-npu` 创建运行目录和缓存。上述用户和路径当前仍是明确固定的部署契约。
PR 会执行仓库测试代码，部署方需使用专用隔离 Runner 并保持 GitHub 的外部贡献者审批策略；不要向测试进程注入部署凭证。

控制台打印本次 `/srv/npu-ci/jobs/slot0/actions-<run_id>-<attempt>.*` 目录。
其中保存 `source.txt`、`checkout/`、独立的 `accuracy/` 和 `performance/` 结果目录及缓存。
阶段目录保存 `context.json`、`report.json`、`summary.md`；算子目录保存原始日志、`process.json`、`result.json` 和已知退出码。
摘要追加到 `GITHUB_STEP_SUMMARY`；JSON 和原始日志保存在虚拟机，尚不上传为 artifacts。运行目录保留核验，清理由部署方安排。
必须从正常基线新建 worktree 部署，不复用故障注入分支。

## 新增用例、算子与事件协议

新增用例：先修改对应脚本的测试计划与实现，再同步 `expected-plans.json` 的参数、顺序、计数和脚本哈希。
新增算子：添加 `ci/npu_ops/<name>/accuracy.py`、`performance.py`，在 manifest 的 `operators` 登记名称、规范脚本路径及数量；在 expected-plans 的 `operators` 登记精度计划、性能计划与 timing。
新增源码依赖时更新 `watch_paths`。公共执行器不增加算子名称分支，不增加 workflow step。

每个结构化事件独占一行，格式为 `NPU_RESULT_JSON=<JSON object>`，使用 `allow_nan=False` 并及时刷新输出。
共同字段为 `schema_version=1`、`operator`、`phase`、`event`；字段和值保持英文机器协议。
普通诊断可使用其他行；不得在普通文字中嵌入该标记。重复键、NaN/Infinity、错序、漏项和计数不符均失败。

- accuracy：`started`（`expected_cases`、`case_names`），每个用例 `case_started`、`case_completed`（完整 `case`、计划中的 metadata 字段以及 `status`），最后 `completed`（预期、已完成、通过、失败计数）。
- performance：`started`（`expected_configs`、`config_names`），每组 `config_started`（完整 `config`、metadata、`warmup`、`iterations`、`expected_samples`）；按零起始 `sample_index` 发出 `sample`（配置名字符串、正且有限的 `latency_ms`）；再发 `config_completed`（采样列表、数量、中位数、`threshold_ms=null`、`performance_verdict=not_evaluated`），最后 `completed`（预期、完成配置计数）。
- 首次失败后立即发 `failed` 终止事件并返回非零，含已完成计数、`current_case` 或 `current_config`、原始 `error_type` 和 `error`；不得继续下一个用例或配置。无法捕获的 import/进程错误允许缺少终止事件，但执行器仍判失败。

具体字段校验以 `reporting.py` 的 `validate_accuracy`、`validate_performance` 为准，可参考两个现有算子脚本。
manifest 或脚本的任意字节变化（包括格式化）都需同步哈希；哈希只证明计划绑定的文件一致，不能替代审核预期值。
在仓库根目录完成格式化后执行：

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('ci/npu_ops')
manifest = json.loads((root / 'manifest.json').read_text())
plan_path = root / 'expected-plans.json'
plan = json.loads(plan_path.read_text())
plan['manifest_sha256'] = hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()
plan['source_sha256'] = {
    path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for entry in manifest['operators'] for path in entry['scripts'].values()
}
plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + '\n')
PY
python3 -B -m unittest discover -s ci/npu_ops/tests -v
```

测试使用临时 Git 仓库、真实子进程和结构化日志，不需要 NPU；它们不能证明真实算子的数值结果。
提交前必须在目标环境执行完整 `pre-commit run --all-files`，复核哈希与回归测试，并使用 signoff。
在已加载环境的 ARM Runner 上，可用同一解释器独立运行某阶段：

```bash
/srv/tokenspeed-npu/cann9-torch29-py312-root/venv/bin/python -B -u \
  ci/npu_ops/run_phase.py --repo-root "$PWD" --phase accuracy \
  --output-dir /tmp/npu-accuracy-unique --timeout-seconds 600
```

直接调用执行器不包含 workflow 的主机锁及部署检查；实际验收优先使用工作流入口，并选择尚不存在的输出目录。
最终还需验证真实 Actions 的成功运行、无关变更跳过、故障精度失败及性能跳过，并核对 Actions 摘要与 VM 文件。

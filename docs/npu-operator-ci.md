# NPU 算子 CI

## 触发与变更判断

工作流为 `.github/workflows/npu-tests.yml`，限定在 `camusmil/tokenspeed` 运行，支持所有分支的 push 和手动触发。
GitHub 托管机器上的 `select` job 比较 push 前后的提交，并按 `ci/npu_ops/manifest.json` 判断变更路径。
命中任一监听路径时，运行全部已登记算子；无关变更跳过 NPU job。新分支或无法完成比较时，保守执行全部算子。
清单或事件输入无效时，检查失败。PR 事件、标签推送和分支删除不运行 NPU 测试。

路径匹配使用区分大小写的 `fnmatchcase`，处理新增、修改、删除及重命名前后的路径，不限制变更文件数量。
kernel/runtime 的宽范围规则可能同时匹配其他后端，但不会修改其他厂商 CI 的判断规则。
检查器输出 `should_run` 和 `reason`，不生成算子矩阵。
[手动触发要求工作流文件存在于默认分支](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_dispatch)。

## 文件职责与调用关系

- `test/ci_system/npu_operator_changes.py`：根据事件和变更路径决定是否运行测试。
- `ci/npu_ops/manifest.json`：登记监听路径、算子顺序、脚本路径和预期数量。
- `ci/npu_ops/expected-plans.json`：记录精度用例、性能配置、采样参数及关联哈希。
- `ci/npu_ops/run_phase.py`：按顺序执行一个阶段，管理子进程、清理资源并返回退出码。
- `ci/npu_ops/reporting.py`：校验结构化日志，生成 JSON 报告和中文 Markdown 摘要。
- `ci/npu_ops/<operator>/accuracy.py`：执行对应算子的精度用例。
- `ci/npu_ops/<operator>/performance.py`：执行对应算子的性能采样。

NPU job 只编写两个 step：全部算子精度测试、全部算子性能测试。GitHub 页面可能另外显示自动准备和清理条目。
精度 step 获取并核对 `GITHUB_SHA` 对应的全新 checkout；性能 step 复用同一份源码，仅在精度 step 成功后运行。
两个 step 分别调用 `run_phase.py`；入口使用自身 Python 解释器启动脚本，并将 checkout 中的两个 kernel 包目录放到 `PYTHONPATH` 前部。

| 算子 | 精度用例数 | 性能配置数 | 性能样本数 |
| --- | --- | --- | --- |
| scatter | 5 | 6 | 30 |
| virtual_slots_to_local | 48 | 8 | 40 |

每组性能配置预热 10 次，随后采集 5 个样本，每个样本测量 50 次执行。
报告中的中位耗时是这 5 个样本各自的单次平均耗时的中位数，单位为毫秒。
当前策略为 `record_only`：`completed` 表示采样完整有效，`performance_verdict` 保持 `not_evaluated`。
尚未配置耗时阈值，不能把采样成功解释为性能达标。

## 失败处理

首个用例、配置、脚本或必要证据保存失败后，停止后续测试；后续算子记为 `not_run`。
精度阶段失败时，整个性能 step 跳过。缺少依赖、日志损坏或不完整、子进程非零退出均使阶段失败。
失败后只继续必要的报告保存和资源清理。Actions 根据入口进程退出码判定 step：0 为成功，非零为失败。
即使测试已经完成，摘要写入失败也会使 step 失败。
当前每个脚本的执行超时为 600 秒，这不是性能达标阈值。
入口收到 SIGINT/SIGTERM 时清理当前测试进程组；中断运行可能只留下部分日志，没有最终报告。

## Runner 环境与报告位置

工作流使用 Runner `tokenspeed-fork-npu-01`、标签 `tokenspeed-fork-npu-setup` 和账号 `ci-npu`。
环境入口为 `/srv/tokenspeed-npu/cann9-torch29-py312-root/env.sh`，临时目录和缓存目录位于本次运行目录内。
每个测试阶段持有 `/srv/npu-ci/locks/host-all.lock`；锁被占用时，在启动测试前失败。

控制台打印位于 `/srv/npu-ci/jobs/slot0/actions-*` 的本次运行目录。
目录内包含 `source.txt`、`checkout/`，以及独立的 `accuracy/`、`performance/` 结果目录。
阶段报告为 `report.json` 和 `summary.md`；已启动算子的目录还保存原始日志、进程信息、结果及已知退出码。
中文摘要在各测试 step 内追加到 `GITHUB_STEP_SUMMARY`，原始日志实时显示在 Actions 中。
JSON 文件保存在虚拟机，当前没有上传为 Actions artifacts。运行目录保留用于核验，需要安排清理。

## 新增算子与验证

在 `ci/npu_ops/<name>/` 下添加精度和性能脚本，在清单中登记规范路径和数量，并在测试计划中补充对应的用例、配置及采样参数。
沿用现有脚本的 `NPU_RESULT_JSON` 事件格式，并在首个失败后立即停止。新增源码根目录时，同时更新监听路径。
清单或测试脚本发生任何修改，包括格式化，都需要更新计划中的 `manifest_sha256` 或对应的 `source_sha256`。
无需增加 workflow step，也无需在通用入口中增加按算子名称分支的逻辑。

在仓库根目录执行 `python -B test/ci_system/test_npu_operator_changes.py -v`，使用临时 Git 仓库运行变更检查器回归测试。
这些 CPU 检查不能替代 NPU 验证。应在 Runner 账号下运行两个真实测试阶段，再通过 Actions 核对工作流成功、失败传播和摘要展示。

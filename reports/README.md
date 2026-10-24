# 报告导出目录

点击右侧“报告”按钮或按 `Ctrl+E` 后会生成：

```text
reports/report_YYYYMMDD_HHMMSS_xxxxxx/
├── match.json
├── summary.json
├── moves.csv
├── mcts_stats.csv
└── screenshot.png
```

`Ctrl+P` 可单独保存当前界面截图到 `reports/screenshots/`。

生成的报告文件默认不纳入 Git，最终需要时可将选定目录复制到课程设计材料中。

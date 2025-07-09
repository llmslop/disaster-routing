#!/bin/jq -rf

(["router", "instance", "mofi", "total_fs"], (.[] | {
  router:.config.router._short_,
  instance: .config.instance.path,
  mofi: first(.run_log[] | select(.msg="Final solution") | .args.mofi),
  total_fs: first(.run_log[] | select(.msg="Final solution") | .args.total_fs),
} | [.router, .instance, .mofi, .total_fs])) | @csv

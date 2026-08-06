#!/bin/zsh -f

emulate zsh

base_commit=9e53613b0ca8e65fbbbb2c8157a0a5e07e497707
bench_commits=$(git log --pretty=%H --grep="^benchmark-line-" ${base_commit}..)
final_commit=21db9ed87af05800f5687bd558a6d8167cbb6b7d
commits=( ${base_commit} ${=bench_commits} ${final_commit} )
pytest-benchmark compare \
    --time-unit=s --sort=fullname --columns=mean,stddev \
    "*"${^commits} \
| awk '
    /^test_json_serialization/ {
        sort_cmd = "sort -gk" (NF + 1);
        if (mean0 == "") {
            mean0 = 0.0 + $3; std0 = 0.0 + $5;
        }
        mean1 = 0.0 + $3; std1 = 0.0 + $5;
        stdtot = sqrt(std0 * std0 + std1 * std1);
        delta = mean1 - mean0;
        rel = delta / stdtot;
        col_mean = NF + 1;
        print $0 "  " delta "  " rel | sort_cmd
        next;
    }
    1 {
        if (sort_cmd != "")  close(sort_cmd);
        sort_cmd = ""
        if (/^Name/) {
            $0 = $0 "  DiffMean  DiffMean/StdTot";
        }
        print $0;
    }
'

#!/bin/zsh -f

emulate zsh

base_commit=18079a5c28a323f9b94de1d1bcb2d9e1053eabe2
bench_commits=$(git log --pretty=%H --grep="^benchmark-line-" ${base_commit}..)
commits=( ${base_commit} ${=bench_commits} )
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

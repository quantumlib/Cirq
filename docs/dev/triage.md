# Cirq Triage

Original RFC: [bit.do/cirq-triage](https://bit.do/cirq-triage)

## **Objective**

The goals for this document are as follows: 

* to define a set of lightweight community processes that make it easy for users and maintainers to understand where certain issues stand, when and how they are going to be resolved, what is blocking them

* provide visibility for project and release status 

## Automation: Triage party and GitHub Actions

[Triage Party](https://github.com/google/triage-party) is a stateless web app to optimize issue and PR triage for large open-source projects using the GitHub API. 

Our deployed version is here (a static IP, domain request is in progress): [http://bit.do/cirq-triage-party](http://bit.do/cirq-triage-party)

[GitHub Actions](https://github.com/features/actions) is GitHub's workflow automation platform. We use it for continuous integration testing as well as for stale issue handling later described here.

## Issue states and labels

### Issue kinds

The following are the kind of issues that Cirq uses: 

* `kind/bug-report` the user found a bug 
* `kind/feature-request` for new functionality 
* `kind/question` in case an issue turns out to be a question, please mark it with kind/question and close it after answering. Also point the user to Quantum Computing Stack Exchange for usage questions and to cirq-dev@googlegroups.com list instead for contribution related questions. 
* `kind/health` For CI/testing/release process/refactoring/technical debt items 
* `kind/docs` documentation problems, ideas, requests
* `kind/roadmap-item` for higher level roadmap items to capture conversations and feedback (not for project tracking)
* `kind/task` for tracking progress on larger efforts 


For most issues there are phases of 

* **triage** do we want to take on this task at all,
* **prioritization** how urgent it is,
* identifying feature **area**,
* signalling **difficulty**,
* signalling **work**,
* **assigning** work,
* and **closing**.

We will explore these phases one by one. 


### Triage

Triage states are 

* `triage/accepted` - there is consensus amongst maintainers that this is a real bug or a reasonable feature to add with a reasonable design, hence it is ready to be implemented.
* `triage/discuss` - can be applied to any of the issue types to bring them up during our weekly Cirq Cynque meeting (join [cirq-dev](https://groups.google.com/g/cirq-dev)) to get an invite!) and/or to signal need for decision. If you mark an issue with triage/discuss, consider pinging the maintainers on the issue who need to come to consensus around the issue. 
* `triage/needs-reproduction` - for bugs only
* `triage/needs-feasibility` - for feature requests (maybe bugs).
* `triage/needs-more-evidence` - for feature requests - the feature request seems plausible but we need more understanding if it is valuable for enough users to warrant implementing and maintaining it. 
* `triage/stale` - GitHub actions automatically marks some of the issues stale and then it closes them in case of 30 days of inactivity.
* `triage/duplicate` - we mark duplicated issues with this label.

While these are fairly straightforward and intuitive the workflows are depicted below. 

#### Bug report triage

![bug workflow](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggVERcblx0QVtVbnRyaWFnZWQgQnVnXSAtLT4gS1xuICAgIEtbbGFiZWw6IGtpbmQvYnVnLXJlcG9ydF0tLT4gQntSZXByb2R1Y2libGU_fVxuXHRCIC0tPiB8bm98IENcbiAgICBCIC0tPiB8eWVzfCBEXG4gICAgQ1tsYWJlbDogdHJpYWdlL25lZWRzLXJlcHJvZHVjdGlvbl0gLS0-IHxhIG1haW50YWluZXIgc3VjY2Vzc2Z1bGx5IHJlcHJvZHVjZXN8IERcbiAgICBDIC0tPiB8bm8gcmVzcG9uc2UgZm9yIDYwZHwgU1xuICAgIFNbbGFiZWw6IHRyaWFnZS9zdGFsZV0gLS0-IEVbQ0xPU0VdXG4gICAgRFtsYWJlbDogdHJpYWdlL2FjY2VwdGVkXSBcbiAgICBcbiAgICBzdHlsZSBEIGZpbGw6IzAwRkYwMCAgICBcbiAgICBzdHlsZSBFIGZpbGw6I0ZGMDAwMCIsIm1lcm1haWQiOnsidGhlbWUiOiJkZWZhdWx0IiwidGhlbWVWYXJpYWJsZXMiOnsiYmFja2dyb3VuZCI6IndoaXRlIiwicHJpbWFyeUNvbG9yIjoiI0VDRUNGRiIsInNlY29uZGFyeUNvbG9yIjoiI2ZmZmZkZSIsInRlcnRpYXJ5Q29sb3IiOiJoc2woODAsIDEwMCUsIDk2LjI3NDUwOTgwMzklKSIsInByaW1hcnlCb3JkZXJDb2xvciI6ImhzbCgyNDAsIDYwJSwgODYuMjc0NTA5ODAzOSUpIiwic2Vjb25kYXJ5Qm9yZGVyQ29sb3IiOiJoc2woNjAsIDYwJSwgODMuNTI5NDExNzY0NyUpIiwidGVydGlhcnlCb3JkZXJDb2xvciI6ImhzbCg4MCwgNjAlLCA4Ni4yNzQ1MDk4MDM5JSkiLCJwcmltYXJ5VGV4dENvbG9yIjoiIzEzMTMwMCIsInNlY29uZGFyeVRleHRDb2xvciI6IiMwMDAwMjEiLCJ0ZXJ0aWFyeVRleHRDb2xvciI6InJnYig5LjUwMDAwMDAwMDEsIDkuNTAwMDAwMDAwMSwgOS41MDAwMDAwMDAxKSIsImxpbmVDb2xvciI6IiMzMzMzMzMiLCJ0ZXh0Q29sb3IiOiIjMzMzIiwibWFpbkJrZyI6IiNFQ0VDRkYiLCJzZWNvbmRCa2ciOiIjZmZmZmRlIiwiYm9yZGVyMSI6IiM5MzcwREIiLCJib3JkZXIyIjoiI2FhYWEzMyIsImFycm93aGVhZENvbG9yIjoiIzMzMzMzMyIsImZvbnRGYW1pbHkiOiJcInRyZWJ1Y2hldCBtc1wiLCB2ZXJkYW5hLCBhcmlhbCIsImZvbnRTaXplIjoiMTZweCIsImxhYmVsQmFja2dyb3VuZCI6IiNlOGU4ZTgiLCJub2RlQmtnIjoiI0VDRUNGRiIsIm5vZGVCb3JkZXIiOiIjOTM3MERCIiwiY2x1c3RlckJrZyI6IiNmZmZmZGUiLCJjbHVzdGVyQm9yZGVyIjoiI2FhYWEzMyIsImRlZmF1bHRMaW5rQ29sb3IiOiIjMzMzMzMzIiwidGl0bGVDb2xvciI6IiMzMzMiLCJlZGdlTGFiZWxCYWNrZ3JvdW5kIjoiI2U4ZThlOCIsImFjdG9yQm9yZGVyIjoiaHNsKDI1OS42MjYxNjgyMjQzLCA1OS43NzY1MzYzMTI4JSwgODcuOTAxOTYwNzg0MyUpIiwiYWN0b3JCa2ciOiIjRUNFQ0ZGIiwiYWN0b3JUZXh0Q29sb3IiOiJibGFjayIsImFjdG9yTGluZUNvbG9yIjoiZ3JleSIsInNpZ25hbENvbG9yIjoiIzMzMyIsInNpZ25hbFRleHRDb2xvciI6IiMzMzMiLCJsYWJlbEJveEJrZ0NvbG9yIjoiI0VDRUNGRiIsImxhYmVsQm94Qm9yZGVyQ29sb3IiOiJoc2woMjU5LjYyNjE2ODIyNDMsIDU5Ljc3NjUzNjMxMjglLCA4Ny45MDE5NjA3ODQzJSkiLCJsYWJlbFRleHRDb2xvciI6ImJsYWNrIiwibG9vcFRleHRDb2xvciI6ImJsYWNrIiwibm90ZUJvcmRlckNvbG9yIjoiI2FhYWEzMyIsIm5vdGVCa2dDb2xvciI6IiNmZmY1YWQiLCJub3RlVGV4dENvbG9yIjoiYmxhY2siLCJhY3RpdmF0aW9uQm9yZGVyQ29sb3IiOiIjNjY2IiwiYWN0aXZhdGlvbkJrZ0NvbG9yIjoiI2Y0ZjRmNCIsInNlcXVlbmNlTnVtYmVyQ29sb3IiOiJ3aGl0ZSIsInNlY3Rpb25Ca2dDb2xvciI6InJnYmEoMTAyLCAxMDIsIDI1NSwgMC40OSkiLCJhbHRTZWN0aW9uQmtnQ29sb3IiOiJ3aGl0ZSIsInNlY3Rpb25Ca2dDb2xvcjIiOiIjZmZmNDAwIiwidGFza0JvcmRlckNvbG9yIjoiIzUzNGZiYyIsInRhc2tCa2dDb2xvciI6IiM4YTkwZGQiLCJ0YXNrVGV4dExpZ2h0Q29sb3IiOiJ3aGl0ZSIsInRhc2tUZXh0Q29sb3IiOiJ3aGl0ZSIsInRhc2tUZXh0RGFya0NvbG9yIjoiYmxhY2siLCJ0YXNrVGV4dE91dHNpZGVDb2xvciI6ImJsYWNrIiwidGFza1RleHRDbGlja2FibGVDb2xvciI6IiMwMDMxNjMiLCJhY3RpdmVUYXNrQm9yZGVyQ29sb3IiOiIjNTM0ZmJjIiwiYWN0aXZlVGFza0JrZ0NvbG9yIjoiI2JmYzdmZiIsImdyaWRDb2xvciI6ImxpZ2h0Z3JleSIsImRvbmVUYXNrQmtnQ29sb3IiOiJsaWdodGdyZXkiLCJkb25lVGFza0JvcmRlckNvbG9yIjoiZ3JleSIsImNyaXRCb3JkZXJDb2xvciI6IiNmZjg4ODgiLCJjcml0QmtnQ29sb3IiOiJyZWQiLCJ0b2RheUxpbmVDb2xvciI6InJlZCIsImxhYmVsQ29sb3IiOiJibGFjayIsImVycm9yQmtnQ29sb3IiOiIjNTUyMjIyIiwiZXJyb3JUZXh0Q29sb3IiOiIjNTUyMjIyIiwiY2xhc3NUZXh0IjoiIzEzMTMwMCIsImZpbGxUeXBlMCI6IiNFQ0VDRkYiLCJmaWxsVHlwZTEiOiIjZmZmZmRlIiwiZmlsbFR5cGUyIjoiaHNsKDMwNCwgMTAwJSwgOTYuMjc0NTA5ODAzOSUpIiwiZmlsbFR5cGUzIjoiaHNsKDEyNCwgMTAwJSwgOTMuNTI5NDExNzY0NyUpIiwiZmlsbFR5cGU0IjoiaHNsKDE3NiwgMTAwJSwgOTYuMjc0NTA5ODAzOSUpIiwiZmlsbFR5cGU1IjoiaHNsKC00LCAxMDAlLCA5My41Mjk0MTE3NjQ3JSkiLCJmaWxsVHlwZTYiOiJoc2woOCwgMTAwJSwgOTYuMjc0NTA5ODAzOSUpIiwiZmlsbFR5cGU3IjoiaHNsKDE4OCwgMTAwJSwgOTMuNTI5NDExNzY0NyUpIn19fQ)

Figure 1. Bug workflow (to edit, see [mermaid source](https://mermaid-js.github.io/mermaid-live-editor/#/edit/eyJjb2RlIjoiZ3JhcGggVERcblx0QVtVbnRyaWFnZWQgQnVnXSAtLT4gS1xuICAgIEtbbGFiZWw6IGtpbmQvYnVnLXJlcG9ydF0tLT4gQntSZXByb2R1Y2libGU_fVxuXHRCIC0tPiB8bm98IENcbiAgICBCIC0tPiB8eWVzfCBEXG4gICAgQ1tsYWJlbDogdHJpYWdlL25lZWRzLXJlcHJvZHVjdGlvbl0gLS0-IHxhIG1haW50YWluZXIgc3VjY2Vzc2Z1bGx5IHJlcHJvZHVjZXN8IERcbiAgICBDIC0tPiB8bm8gcmVzcG9uc2UgZm9yIDYwZHwgU1xuICAgIFNbbGFiZWw6IHRyaWFnZS9zdGFsZV0gLS0-IEVbQ0xPU0VdXG4gICAgRFtsYWJlbDogdHJpYWdlL2FjY2VwdGVkXSBcbiAgICBcbiAgICBzdHlsZSBEIGZpbGw6IzAwRkYwMCAgICBcbiAgICBzdHlsZSBFIGZpbGw6I0ZGMDAwMCIsIm1lcm1haWQiOnsidGhlbWUiOiJkZWZhdWx0In19))

#### Feature request triage

![feature request workflow](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggVERcblx0QVtVbnRyaWFnZWQgRmVhdHVyZSBSZXF1ZXN0XSAtLT4gS1xuICAgIEtbbGFiZWw6IGtpbmQvZmVhdHVyZS1yZXF1ZXN0XSAtLT4gTkR7TmVlZHMgZGlzY3Vzc2lvbj99XG4gICAgTkQgLS0-IHx5ZXN8IEZbbGFiZWw6IHRyaWFnZS9kaXNjdXNzICYgQ2lycSBDeW5xdWVdICAgICBcbiAgICBORCAtLT4gfG5vfCBCe1Nob3VsZCB3ZSBkbyB0aGlzP31cbiAgICBGIC0tPiBCXG5cdEIgLS0-IHxub3wgRVtDTE9TRV1cbiAgICBCIC0tPiB8eWVzfCBEXG4gICAgQiAtLT4gfHdlIGFyZSBub3Qgc3VyZSBpdCdzIGZlYXNpYmxlIHwgRFBbbGFiZWw6IHRyaWFnZS9uZWVkcy1mZWFzaWJpbGl0eV0gIFxuICAgIEIgLS0-IHxtYXliZSBpZiBlbm91Z2ggcGVvcGxlIHJlcXVlc3QgaXR8IEVWSVtsYWJlbDogdHJpYWdlL2F3YWl0aW5nLW1vcmUtZXZpZGVuY2VdXG4gICAgRFAgLS0-IHxkZXNpZ24gcHJvcG9zZWQvaW1wcm92ZWR8IE5EICAgICBcbiAgICBEW2xhYmVsOiB0cmlhZ2UvYWNjZXB0ZWRdXG4gICAgRFAgLS0-IHxzdGFsZSBhZnRlciA2MGR8RVxuICAgIEVWSSAtLT4gfHN0YWxlIGFmdGVyIDYwZHxFXG5cdFxuXG4gICAgc3R5bGUgRCBmaWxsOiMwMEZGMDAgICAgICAgXG4gICAgc3R5bGUgRSBmaWxsOiNGRjAwMDAiLCJtZXJtYWlkIjp7InRoZW1lIjoiZGVmYXVsdCIsInRoZW1lVmFyaWFibGVzIjp7ImJhY2tncm91bmQiOiJ3aGl0ZSIsInByaW1hcnlDb2xvciI6IiNFQ0VDRkYiLCJzZWNvbmRhcnlDb2xvciI6IiNmZmZmZGUiLCJ0ZXJ0aWFyeUNvbG9yIjoiaHNsKDgwLCAxMDAlLCA5Ni4yNzQ1MDk4MDM5JSkiLCJwcmltYXJ5Qm9yZGVyQ29sb3IiOiJoc2woMjQwLCA2MCUsIDg2LjI3NDUwOTgwMzklKSIsInNlY29uZGFyeUJvcmRlckNvbG9yIjoiaHNsKDYwLCA2MCUsIDgzLjUyOTQxMTc2NDclKSIsInRlcnRpYXJ5Qm9yZGVyQ29sb3IiOiJoc2woODAsIDYwJSwgODYuMjc0NTA5ODAzOSUpIiwicHJpbWFyeVRleHRDb2xvciI6IiMxMzEzMDAiLCJzZWNvbmRhcnlUZXh0Q29sb3IiOiIjMDAwMDIxIiwidGVydGlhcnlUZXh0Q29sb3IiOiJyZ2IoOS41MDAwMDAwMDAxLCA5LjUwMDAwMDAwMDEsIDkuNTAwMDAwMDAwMSkiLCJsaW5lQ29sb3IiOiIjMzMzMzMzIiwidGV4dENvbG9yIjoiIzMzMyIsIm1haW5Ca2ciOiIjRUNFQ0ZGIiwic2Vjb25kQmtnIjoiI2ZmZmZkZSIsImJvcmRlcjEiOiIjOTM3MERCIiwiYm9yZGVyMiI6IiNhYWFhMzMiLCJhcnJvd2hlYWRDb2xvciI6IiMzMzMzMzMiLCJmb250RmFtaWx5IjoiXCJ0cmVidWNoZXQgbXNcIiwgdmVyZGFuYSwgYXJpYWwiLCJmb250U2l6ZSI6IjE2cHgiLCJsYWJlbEJhY2tncm91bmQiOiIjZThlOGU4Iiwibm9kZUJrZyI6IiNFQ0VDRkYiLCJub2RlQm9yZGVyIjoiIzkzNzBEQiIsImNsdXN0ZXJCa2ciOiIjZmZmZmRlIiwiY2x1c3RlckJvcmRlciI6IiNhYWFhMzMiLCJkZWZhdWx0TGlua0NvbG9yIjoiIzMzMzMzMyIsInRpdGxlQ29sb3IiOiIjMzMzIiwiZWRnZUxhYmVsQmFja2dyb3VuZCI6IiNlOGU4ZTgiLCJhY3RvckJvcmRlciI6ImhzbCgyNTkuNjI2MTY4MjI0MywgNTkuNzc2NTM2MzEyOCUsIDg3LjkwMTk2MDc4NDMlKSIsImFjdG9yQmtnIjoiI0VDRUNGRiIsImFjdG9yVGV4dENvbG9yIjoiYmxhY2siLCJhY3RvckxpbmVDb2xvciI6ImdyZXkiLCJzaWduYWxDb2xvciI6IiMzMzMiLCJzaWduYWxUZXh0Q29sb3IiOiIjMzMzIiwibGFiZWxCb3hCa2dDb2xvciI6IiNFQ0VDRkYiLCJsYWJlbEJveEJvcmRlckNvbG9yIjoiaHNsKDI1OS42MjYxNjgyMjQzLCA1OS43NzY1MzYzMTI4JSwgODcuOTAxOTYwNzg0MyUpIiwibGFiZWxUZXh0Q29sb3IiOiJibGFjayIsImxvb3BUZXh0Q29sb3IiOiJibGFjayIsIm5vdGVCb3JkZXJDb2xvciI6IiNhYWFhMzMiLCJub3RlQmtnQ29sb3IiOiIjZmZmNWFkIiwibm90ZVRleHRDb2xvciI6ImJsYWNrIiwiYWN0aXZhdGlvbkJvcmRlckNvbG9yIjoiIzY2NiIsImFjdGl2YXRpb25Ca2dDb2xvciI6IiNmNGY0ZjQiLCJzZXF1ZW5jZU51bWJlckNvbG9yIjoid2hpdGUiLCJzZWN0aW9uQmtnQ29sb3IiOiJyZ2JhKDEwMiwgMTAyLCAyNTUsIDAuNDkpIiwiYWx0U2VjdGlvbkJrZ0NvbG9yIjoid2hpdGUiLCJzZWN0aW9uQmtnQ29sb3IyIjoiI2ZmZjQwMCIsInRhc2tCb3JkZXJDb2xvciI6IiM1MzRmYmMiLCJ0YXNrQmtnQ29sb3IiOiIjOGE5MGRkIiwidGFza1RleHRMaWdodENvbG9yIjoid2hpdGUiLCJ0YXNrVGV4dENvbG9yIjoid2hpdGUiLCJ0YXNrVGV4dERhcmtDb2xvciI6ImJsYWNrIiwidGFza1RleHRPdXRzaWRlQ29sb3IiOiJibGFjayIsInRhc2tUZXh0Q2xpY2thYmxlQ29sb3IiOiIjMDAzMTYzIiwiYWN0aXZlVGFza0JvcmRlckNvbG9yIjoiIzUzNGZiYyIsImFjdGl2ZVRhc2tCa2dDb2xvciI6IiNiZmM3ZmYiLCJncmlkQ29sb3IiOiJsaWdodGdyZXkiLCJkb25lVGFza0JrZ0NvbG9yIjoibGlnaHRncmV5IiwiZG9uZVRhc2tCb3JkZXJDb2xvciI6ImdyZXkiLCJjcml0Qm9yZGVyQ29sb3IiOiIjZmY4ODg4IiwiY3JpdEJrZ0NvbG9yIjoicmVkIiwidG9kYXlMaW5lQ29sb3IiOiJyZWQiLCJsYWJlbENvbG9yIjoiYmxhY2siLCJlcnJvckJrZ0NvbG9yIjoiIzU1MjIyMiIsImVycm9yVGV4dENvbG9yIjoiIzU1MjIyMiIsImNsYXNzVGV4dCI6IiMxMzEzMDAiLCJmaWxsVHlwZTAiOiIjRUNFQ0ZGIiwiZmlsbFR5cGUxIjoiI2ZmZmZkZSIsImZpbGxUeXBlMiI6ImhzbCgzMDQsIDEwMCUsIDk2LjI3NDUwOTgwMzklKSIsImZpbGxUeXBlMyI6ImhzbCgxMjQsIDEwMCUsIDkzLjUyOTQxMTc2NDclKSIsImZpbGxUeXBlNCI6ImhzbCgxNzYsIDEwMCUsIDk2LjI3NDUwOTgwMzklKSIsImZpbGxUeXBlNSI6ImhzbCgtNCwgMTAwJSwgOTMuNTI5NDExNzY0NyUpIiwiZmlsbFR5cGU2IjoiaHNsKDgsIDEwMCUsIDk2LjI3NDUwOTgwMzklKSIsImZpbGxUeXBlNyI6ImhzbCgxODgsIDEwMCUsIDkzLjUyOTQxMTc2NDclKSJ9fX0)

Figure 2. Feature request workflow (to edit, see [mermaid source](https://mermaid-js.github.io/mermaid-live-editor/#/edit/eyJjb2RlIjoiZ3JhcGggVERcblx0QVtVbnRyaWFnZWQgRmVhdHVyZSBSZXF1ZXN0XSAtLT4gS1xuICAgIEtbbGFiZWw6IGtpbmQvZmVhdHVyZS1yZXF1ZXN0XSAtLT4gTkR7TmVlZHMgZGlzY3Vzc2lvbj99XG4gICAgTkQgLS0-IHx5ZXN8IEZbbGFiZWw6IHRyaWFnZS9kaXNjdXNzICYgQ2lycSBDeW5xdWVdICAgICBcbiAgICBORCAtLT4gfG5vfCBCe1Nob3VsZCB3ZSBkbyB0aGlzP31cbiAgICBGIC0tPiBCXG5cdEIgLS0-IHxub3wgRVtDTE9TRV1cbiAgICBCIC0tPiB8eWVzfCBEXG4gICAgQiAtLT4gfHdlIGFyZSBub3Qgc3VyZSBpdCdzIGZlYXNpYmxlIHwgRFBbbGFiZWw6IHRyaWFnZS9uZWVkcy1mZWFzaWJpbGl0eV0gIFxuICAgIEIgLS0-IHxtYXliZSBpZiBlbm91Z2ggcGVvcGxlIHJlcXVlc3QgaXR8IEVWSVtsYWJlbDogdHJpYWdlL2F3YWl0aW5nLW1vcmUtZXZpZGVuY2VdXG4gICAgRFAgLS0-IHxkZXNpZ24gcHJvcG9zZWQvaW1wcm92ZWR8IE5EICAgICBcbiAgICBEW2xhYmVsOiB0cmlhZ2UvYWNjZXB0ZWRdXG4gICAgRFAgLS0-IHxzdGFsZSBhZnRlciA2MGR8RVxuICAgIEVWSSAtLT4gfHN0YWxlIGFmdGVyIDYwZHxFXG5cdFxuXG4gICAgc3R5bGUgRCBmaWxsOiMwMEZGMDAgICAgICAgXG4gICAgc3R5bGUgRSBmaWxsOiNGRjAwMDAiLCJtZXJtYWlkIjp7InRoZW1lIjoiZGVmYXVsdCIsInRoZW1lVmFyaWFibGVzIjp7ImJhY2tncm91bmQiOiJ3aGl0ZSIsInByaW1hcnlDb2xvciI6IiNFQ0VDRkYiLCJzZWNvbmRhcnlDb2xvciI6IiNmZmZmZGUiLCJ0ZXJ0aWFyeUNvbG9yIjoiaHNsKDgwLCAxMDAlLCA5Ni4yNzQ1MDk4MDM5JSkiLCJwcmltYXJ5Qm9yZGVyQ29sb3IiOiJoc2woMjQwLCA2MCUsIDg2LjI3NDUwOTgwMzklKSIsInNlY29uZGFyeUJvcmRlckNvbG9yIjoiaHNsKDYwLCA2MCUsIDgzLjUyOTQxMTc2NDclKSIsInRlcnRpYXJ5Qm9yZGVyQ29sb3IiOiJoc2woODAsIDYwJSwgODYuMjc0NTA5ODAzOSUpIiwicHJpbWFyeVRleHRDb2xvciI6IiMxMzEzMDAiLCJzZWNvbmRhcnlUZXh0Q29sb3IiOiIjMDAwMDIxIiwidGVydGlhcnlUZXh0Q29sb3IiOiJyZ2IoOS41MDAwMDAwMDAxLCA5LjUwMDAwMDAwMDEsIDkuNTAwMDAwMDAwMSkiLCJsaW5lQ29sb3IiOiIjMzMzMzMzIiwidGV4dENvbG9yIjoiIzMzMyIsIm1haW5Ca2ciOiIjRUNFQ0ZGIiwic2Vjb25kQmtnIjoiI2ZmZmZkZSIsImJvcmRlcjEiOiIjOTM3MERCIiwiYm9yZGVyMiI6IiNhYWFhMzMiLCJhcnJvd2hlYWRDb2xvciI6IiMzMzMzMzMiLCJmb250RmFtaWx5IjoiXCJ0cmVidWNoZXQgbXNcIiwgdmVyZGFuYSwgYXJpYWwiLCJmb250U2l6ZSI6IjE2cHgiLCJsYWJlbEJhY2tncm91bmQiOiIjZThlOGU4Iiwibm9kZUJrZyI6IiNFQ0VDRkYiLCJub2RlQm9yZGVyIjoiIzkzNzBEQiIsImNsdXN0ZXJCa2ciOiIjZmZmZmRlIiwiY2x1c3RlckJvcmRlciI6IiNhYWFhMzMiLCJkZWZhdWx0TGlua0NvbG9yIjoiIzMzMzMzMyIsInRpdGxlQ29sb3IiOiIjMzMzIiwiZWRnZUxhYmVsQmFja2dyb3VuZCI6IiNlOGU4ZTgiLCJhY3RvckJvcmRlciI6ImhzbCgyNTkuNjI2MTY4MjI0MywgNTkuNzc2NTM2MzEyOCUsIDg3LjkwMTk2MDc4NDMlKSIsImFjdG9yQmtnIjoiI0VDRUNGRiIsImFjdG9yVGV4dENvbG9yIjoiYmxhY2siLCJhY3RvckxpbmVDb2xvciI6ImdyZXkiLCJzaWduYWxDb2xvciI6IiMzMzMiLCJzaWduYWxUZXh0Q29sb3IiOiIjMzMzIiwibGFiZWxCb3hCa2dDb2xvciI6IiNFQ0VDRkYiLCJsYWJlbEJveEJvcmRlckNvbG9yIjoiaHNsKDI1OS42MjYxNjgyMjQzLCA1OS43NzY1MzYzMTI4JSwgODcuOTAxOTYwNzg0MyUpIiwibGFiZWxUZXh0Q29sb3IiOiJibGFjayIsImxvb3BUZXh0Q29sb3IiOiJibGFjayIsIm5vdGVCb3JkZXJDb2xvciI6IiNhYWFhMzMiLCJub3RlQmtnQ29sb3IiOiIjZmZmNWFkIiwibm90ZVRleHRDb2xvciI6ImJsYWNrIiwiYWN0aXZhdGlvbkJvcmRlckNvbG9yIjoiIzY2NiIsImFjdGl2YXRpb25Ca2dDb2xvciI6IiNmNGY0ZjQiLCJzZXF1ZW5jZU51bWJlckNvbG9yIjoid2hpdGUiLCJzZWN0aW9uQmtnQ29sb3IiOiJyZ2JhKDEwMiwgMTAyLCAyNTUsIDAuNDkpIiwiYWx0U2VjdGlvbkJrZ0NvbG9yIjoid2hpdGUiLCJzZWN0aW9uQmtnQ29sb3IyIjoiI2ZmZjQwMCIsInRhc2tCb3JkZXJDb2xvciI6IiM1MzRmYmMiLCJ0YXNrQmtnQ29sb3IiOiIjOGE5MGRkIiwidGFza1RleHRMaWdodENvbG9yIjoid2hpdGUiLCJ0YXNrVGV4dENvbG9yIjoid2hpdGUiLCJ0YXNrVGV4dERhcmtDb2xvciI6ImJsYWNrIiwidGFza1RleHRPdXRzaWRlQ29sb3IiOiJibGFjayIsInRhc2tUZXh0Q2xpY2thYmxlQ29sb3IiOiIjMDAzMTYzIiwiYWN0aXZlVGFza0JvcmRlckNvbG9yIjoiIzUzNGZiYyIsImFjdGl2ZVRhc2tCa2dDb2xvciI6IiNiZmM3ZmYiLCJncmlkQ29sb3IiOiJsaWdodGdyZXkiLCJkb25lVGFza0JrZ0NvbG9yIjoibGlnaHRncmV5IiwiZG9uZVRhc2tCb3JkZXJDb2xvciI6ImdyZXkiLCJjcml0Qm9yZGVyQ29sb3IiOiIjZmY4ODg4IiwiY3JpdEJrZ0NvbG9yIjoicmVkIiwidG9kYXlMaW5lQ29sb3IiOiJyZWQiLCJsYWJlbENvbG9yIjoiYmxhY2siLCJlcnJvckJrZ0NvbG9yIjoiIzU1MjIyMiIsImVycm9yVGV4dENvbG9yIjoiIzU1MjIyMiIsImNsYXNzVGV4dCI6IiMxMzEzMDAiLCJmaWxsVHlwZTAiOiIjRUNFQ0ZGIiwiZmlsbFR5cGUxIjoiI2ZmZmZkZSIsImZpbGxUeXBlMiI6ImhzbCgzMDQsIDEwMCUsIDk2LjI3NDUwOTgwMzklKSIsImZpbGxUeXBlMyI6ImhzbCgxMjQsIDEwMCUsIDkzLjUyOTQxMTc2NDclKSIsImZpbGxUeXBlNCI6ImhzbCgxNzYsIDEwMCUsIDk2LjI3NDUwOTgwMzklKSIsImZpbGxUeXBlNSI6ImhzbCgtNCwgMTAwJSwgOTMuNTI5NDExNzY0NyUpIiwiZmlsbFR5cGU2IjoiaHNsKDgsIDEwMCUsIDk2LjI3NDUwOTgwMzklKSIsImZpbGxUeXBlNyI6ImhzbCgxODgsIDEwMCUsIDkzLjUyOTQxMTc2NDclKSJ9fSwidXBkYXRlRWRpdG9yIjpmYWxzZX0))

#### Other issue types

For `kind/docs`, the label `triage/accepted` has to be added by at least one of the maintainers. 

For `kind/health`, `kind/roadmap-item` and `kind/task` there is no particular intake workflow, as we assume that only maintainers create them to track specific work items.

### Prioritization 

Labels for priority capture the **community's intent** around when a certain feature/bug/task should be done by. It is decided by the Triage team, based on the negotiation with the user who opened the issue. Priority is expected to be modified throughout the lifetime of the issue as the expectations evolve around it. 

- `priority/p0` should be very rare, only cases of emergency, and when a major critical user journey is blocked (e.g. users are exposed to a security vulnerability or they can't install Cirq)
- `priority/p1` is reserved for issues that need to be addressed for high priority work (e.g. a publication that is planned earlier than the next release) 
- `priority/p2`  and `priority/p3` are used to tie into release planning conversations and to signal contributors important work that can be picked up. 

Features and Bugs with no priority label on them will still be up for grabs for contributors. Community contributors assigned to an issue that has no priority have the discretion to choose which release they will finish the issue by.

### Labels for feature area

The goal of [feature area labels](https://github.com/quantumlib/Cirq/labels?q=area) are to enable easy filtering to certain areas. This can help during planning, exploring problematic areas, and finding duplicate issues. Multiple `area/*` labels can be added to a single issue. 


### Signalling difficulty

Difficulty is a function of 

* complexity - the size/hardness of the issue 
* the skills required by the issue and the contributor's skills 

**Complexity**

* `complexity/low` - involves introducing/modifying less than 1-2 concepts, should take 1-2 days max for an advanced contributor
* `complexity/medium` - involves introducing/modifying 3-5 concepts, takes max up to a month for an advanced contributor
* `complexity/high` - involves introducing/modifying 6+ concepts, can take more than a month for an advanced contributor to work through it, and/or modifies core concepts in Cirq 

**Skill level required** (skill/<level>)

* `none`: no special background knowledge required
* `beginner`: little to no background knowledge is required in the given `area/*` labels
* `advanced`: requires solid understanding at least one of the areas signalled by the `area/*` labels
* `expert`: requires deep insight about one or more `area/*` labels to design the right abstractions

### Signalling work for contributors

* `good first issue`: (`level/beginner` in the areas needed and `complexity/low` to `complexity/medium`) - the issue is relatively small, self contained, doesn't require too much QC knowledge  
* `good for learning`: (`level/advanced` in the areas needed and `complexity/low`) - the issue is relatively small, self contained, but requires digging into some areas and develop a solid understanding. Should be a bit harder than "good first issues". 
* `good part time project` - (`level/advanced` and `complexity/medium`) - the issue might take up a couple of months, needs a design and multiple conversations, can require digging deep into a couple of papers. It is still self-contained, doesn't have too much dependencies on the rest of Cirq. 
* `help wanted` - If a project lead wants help on a certain task or a high priority item needs to be done but no one is assigned to it yet, we should put the `help wanted` label on it.  

### Implementation and design

After an issue arrives to `triage/accepted` there can be two avenues: it is ready to be implemented (most of the cases) or it needs design work upfront. 

When an issue is ready to be implemented, no extra label is required to signal the readiness, that is the default. 

However, when there is a need for design, we add the label `needs agreed design`. The design could be as lightweight as a discussion in the issue itself or a full fledged [RFC proposal](rfc_process.md) which should be clear from the comments. 

### Assigning work

Assignment should be a function of 

* **willingness** - contributors should volunteer to take issues or maintainers should take them actively.
* **priority** - critical issues shouldn't depend on part time work.
* **complexity** - highly complex, large pieces are not feasible/rewarding part time necessarily.
* **skills** - if someone does not have the skills for a given issue, they will have to factor in the learning that's required to do it.

### Closing

Issues should be automatically closed by PRs using the `Fixes #XYZD.` phrase in their description or manually, referring to the PR in case the PR author forgot to add the phrase. 

### Stale issues


- Bugs and Feature requests in states `triage/needs-reproduction` and `triage/needs-design-work`, i.e. where the author is required to provide more details get an automated comment "*This issue has not received any updates in 30 days*" and then is marked as `triage/stale` after 60 days and are closed. 
- Documentation (`kind/docs`) issues **without** `triage/accepted` or `triage/discuss` are subject to 60 days staleness policy as well. 
- Roadmap-items and Tasks, and issues in `triage/accepted` or `triage/discuss` state never get stale automatically, they are subject to review during daily / weekly triage and the twice a year **Bug Smash**.

To summarize, **all issues** are subject to staleness-check, **except** the following:

* `triage/accepted`
* `triage/discuss`
* `kind/health`
* `kind/roadmap-item`
* `kind/task` 

The staleness check automation is implemented via GitHub Actions, the latest definition of staleness is defined in [our staleness GitHub Action workflow](https://github.com/quantumlib/Cirq/blob/main/.github/workflows/stale.yml).


## Processes

### Daily triage

**Goals:**

* P0 - notice high priority issues as soon as possible and organize a fix for them.
* P1 - keep the issue and PR backlog clean  
    - maintain a backlog that makes it easy to match contributors as well as maintainers to work items.
    - for pull requests we are aiming for 
        * **responsiveness** - people can get their work done - we don't want to block community / our team members.
        * **clean workspace** - stale PRs are wasteful as clutter is cognitive cost for maintainers. Stale PRs also a resource cost on GitHub - eating into other contributors' capacity to execute GitHub Actions / checks.

**Who**

* [mandatory] Cirq maintainers on weekly Cirq rotation - key thing is to cover p0 bugs.
* [optional] any maintainer who has Triage access rights to the repo. 

**When**

* daily, continuously  - Cirq maintainer rotation is weekly

**What**

Issues: Daily triage should make sure that each issue has the following labels: 

* `triage/*`
* `area/*`
* `complexity/*`
* `skill/*`

Pull requests: 

* As a triager it is your responsibility to review as many PRs as possible during your triage week. 

### Weekly discussions

**Goals**

* make design decisions together with the maintainers on items that need to be discussed
* provide a forum for feedback and blockages 
* plan together features and releases as a community 

**Who**

* everyone on the [cirq-dev](https://groups.google.com/g/cirq-dev) email list is invited 

**When:**

* 11:00AM-12:00PM PST Wednesdays 

**What:**

Cirq Cynque should be the place to discuss 

* as much of the `triage/discuss` items as possible and make decisions about controversial bugs and feature requests. 
* **prioritization requests** - stakeholders, like quantum platform providers, research teams should be able to advocate for raising the priority of certain items
* **release planning / status**: only issues with owners should be added to milestones. The owners are responsible to notify the maintainers in case the issue won't be resolved until the release.

### Bug smash - every 6 months

**Goals:**
* keep the triage alive: catch up on untriaged issues 
* keep the backlog of issues clean and relevant
* use the outstanding backlog as the driver for roadmap planning 

**Who:**

* core maintainers

**When:**

* every 6 months

**What:**

Every 6 months, after every other release, the team should come together and review `triage/accepted` items and revisit them. This is also a chance to catchup on daily triage in case it slipped. 

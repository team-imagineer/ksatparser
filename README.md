## 사용법
```python
import ksatparser

pdf_path = "your_pdf_path"
output_dir = "your_output_dir"

ksatparser.parse_problem(pdf_path, output_dir) # 기출문제 파싱
ksatparser.parse_solution(pdf_path, output_dir) # 해설지 파싱
```
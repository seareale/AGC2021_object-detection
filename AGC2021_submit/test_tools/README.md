# Example

```python
from f1score import *

true_dict = get_true_annotation('../seareale/config.yaml')
pred_dict = get_pred_annotation('../seareale/answersheet_4_03_seareale.json')

macro_f1, all_f1_list, conf_mat_list  = AGC2021_f1score(true_dict, pred_dict)
macro_f1, all_f1_list, conf_mat_list  = AGC2021_f1score(true_dict, pred_dict, include_zero=True)
```
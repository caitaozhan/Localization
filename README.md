# Localization

Multiple transmitter localization

## full training
  NUM INTRU    our error    splot error
-----------  -----------  -------------
          1     0.497692       0.9175              # our error needs to go up a little
          2     1.30924        1.49604             # needs to keep going down (by two Tx far away)
          3     0.714796       0.799704 

  NUM INTRU    our miss    splot miss
-----------  ----------  ------------
          1   0.0892857      0.160714              # splot miss needs to go down a little
          2   0.0434783      0.152174
          3   0.111          0.222111 

  NUM INTRU    our false_alarm    splot false_alarm
-----------  -----------------  -------------------
          1          0.0803571             0.116071
          2          0.0652174             0.108696
          3          0.0925                0.129556


## with interpolation
  NUM INTRU    our error    splot error
-----------  -----------  -------------
          1     0.809217       0.914696
          2     0.90525        1.2805 

  NUM INTRU    our miss    splot miss
-----------  ----------  ------------
          1        0.08         0.08
          2        0            0.125 

  NUM INTRU    our false_alarm    splot false_alarm
-----------  -----------------  -------------------
          1             0.06                  0.12
          2             0.0625                0.125
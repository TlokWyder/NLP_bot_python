import re


born = "05-03-1987 # Дата рождения"

# Удалим комментарий из строки
dob = re.sub(r'#.*$', ".", born)
print("Дата рождения:", dob)

# Заменим дефисы на точки
f_dob = re.sub(r'-', "*", born)
print(f_dob)
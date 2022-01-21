'''
python 一切皆对象
'''


# 函数赋值给变量
def say(text):
    print(text)


speak = say
speak('hello')
# del say
# say('hello')
speak('hello')

# 函数作为变量，存储到数据结构
funcs = [say, str.lower, str.upper]
for func in funcs:
    print(func, func('hello'))

# 函数作为参数传递
print('-' * 20)


def sayUpper(text):
    print(text.upper())


def sayLower(text):
    print(text.lower())


def speak(text, func):
    func(text)


speak('hello', sayLower)
speak('hello', sayUpper)

# 函数作为返回值
print('-' * 20)


def speaker(volume):
    def quiet(text):
        return text.lower()

    def loud(text):
        return text.upper()

    if volume < 20:
        return quiet
    else:
        return loud


s = speaker(10)
print(s('hello'))
S = speaker(30)
print(S('hello'))


# 像执行函数一样执行对象
class Subtrack:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, x):
        return self.a - self.b - x


print('-' * 20)

X = Subtrack(10, 5)
print(X(2))

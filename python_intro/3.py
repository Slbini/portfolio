#3강

#자료형이란?
#파이썬에서 데이터를 다룰 때 데이터의 종류를 의미
#변수를 만들 때 사용자가 자료형을 결정하지 않아도 파이썬 내부에서 자동으로 자료형을 판단하여 적용된다
#자료형 확인은 type()함수로 알 수 있다.
#필요에 따라 자료형을 변경할 수 있다
#각 자료형의 특징을 잘 이해하면 효율적인 코드를 짤 수 있다



#1. 숫자형

#정수형int
#실수형 float
#복소수 complex 등등

#연산이 가능
#숫자를 다루는 내장함수들 사용가능 ex)round(), range(), pow() 등

#a//b : a를 b로 나눈 몫
#a%b: a를 b로 나눈 나머지
#a**b: a의 b제곱

a=10
b=5
c=2.0
d=0.5

print(a+b)
print(a-b)
print(a*b)
print(a/b)
print(a//b)
print(a%b)
print(a**b)

#int와 int의 연산의 결과는 int로 나오고 나누기를 했을 때만 float로 나옴
#a/b와 a//b는 모두 값이 2 이지만, a/b는 2.0, a//b는 2로 나옴

print(b+c)
print(b-c)
print(b*c)
print(b/c)
print(b//c)
print(b%c)
print(b**c)

#int와 float의 결과는 모두 float의 형태

#float끼리의 결과는 모두 float의 형태



#논리형

#[종류] bool -> true참/false거짓

#참과 거짓을 나타내는 자료형
#주로 비교&논리 연산자로 만들어진다
#조건문에 많이 활용된다

#비교연산자
#<작다
#<=작거나 같다
#>크다
#>=크거나 같다
## ==같다
### =은 대입하겠다는 의미!!  오른쪽 값을 왼쪽에 집어 넣겠다는 뜻
##!=같지 않다

#논리 연산자
#x or y: x나 y 둘 중 하나만 참이면 참
#x and y: x,y 모두 참이어야 참
#not x: x가 참이면 거짓 x가 거짓이면 참

x=10
y=-10

print(x>10)
print(y>0)
print()
print(x>y)
print(x<y)
print()
print(x==10)
print(x==y)
print(x!=y)
print()
print(x>0 or y>0)
print(x>0 and y>0)
print()
print(x>0)
print(not x>0)




#문자열형

#[종류] str -> 다른 언어와 달리 문자와 문자열을 따로 구분하지 않는다
#''또는 ""에 감싸져 있다
#연산이 불가능하다 (예외: 문자+문자, 문자*정수)
#문자열을 다루는 다양한 메소드들이 존재한다
#ex) split(), join(), upper(), lower(), replace()

a=5 #int
b='5' #str
c=5.0 #float

print(a+a) #int + int
print()
print(b+b) #str + str #문자가 그냥 이어붙어짐 #55
print(a*b) #int * str #문자가 곱하기된 숫자만큼 이어짐 #55555
print()
"""
print(a+b) #int + str
print(b*c) #str*float
"""

print('안녕하세요')
print("안녕하세요")
#print("안녕하세요')



#군집 자료형
#[종류]   
#여러 데이터를 모은 집합 형태 자료형

##리스트 list  #데이터를 순차적으로 저장->열거형 
##튜플 tuple  #값을 변경할 수 없는 열거형 집합 
##세트 set  #순서가 없고 중복이 허용되지 않는 집합
##사전 dictionary  #키와 값의 쌍으로 구성된 집합



#자료형 변환

#파이썬은 사용자가 자료형을 결정하지 않기 때문에 편리하기도 하지만 데이터가 사용자의 의도와 다른 자료형이 될 수도 있다
#이때는 각 데이터들의 자료형을 원하는 것으로 변경해야 한다
#ex) input() 사용, 정수와 실수의 사용 등

#자료형 변환(typecasting)방법: 원하는 자료형 함수에 값을 넣는다
#ex) str(10), int('10'), int(12.5),, list('hello')


##실습 # input()으로 숫자 입력 받기
'''
a=int(input('숫자를 입력하세요.'))

print(a+a)
'''


"""
# 실수 <-> 정수
num=5.0
range(int(num))
"""

'''
a=input('숫자 하나 입력')
b=int(a)
c=float(a)

print(type(a)) #str
print(type(b)) #int
print(type(c)) #float
'''




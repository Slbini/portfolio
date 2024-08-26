#연산자 우선 순위
#산술 연산자>관계 연산자>논리 연산자

#논리 연산자
#not>and>or

#ex)2*5>2+5 and not 3*3>10 -> True




#반올림:round(a), round(a,b)
#소수 b번째 자리까지 a를 반올림
"""
print(round(3.33))
print(round(3.66))
print(round(3.66,1))
"""


#절대값:abs(a)
"""
print(abs(3))
print(abs(-3))
"""


# 제곱:pow(a,b)
#a:밑 #b:지수
"""
print(pow(3,2))
print(3**2)
"""



#나눗셈:divmod(a,b)
#//:몫 반환  #% 나머지 반환
#divmod: 몫과 나머지 모두 반환 # 변수 두 개
'''
x,y=divmod(7,2)
print(x)
print(y)
'''



'''
#최대값: max(a,b,c,d,,,,)
print(max([7,5,1,3]))

#최소값: min(a,b,c,d,,,,)
print(min([7,5,1,3]))
'''



#합: sum(집합형태 :iterable)
print(sum([7,5,1,3])) #모아 놓은 형태만 들어갈 수 있음

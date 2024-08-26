'''
#한 줄에 하나씩 숫자 입력받기
a=int(input('a 입력:'))
b=int(input('b 입력:'))
c=int(input('c 입력:'))

print(a,b,c,a+b+c)
'''

'''
#한 줄에 여러 개의 숫자 입력받기
a,b,c=map(int,input('a b c 값 입력').split())
#input을 받아서 split으로 공백 기준 자르기
#int로 바꿔주기


print(a,b,c,a+b+c)
'''

#함수는 그냥 사용할 수 있지만,
#method는 사용할 때 앞에 내가 사용하는 대상이 '.'으로 연결되어 있어야


'''
#문자1.split(문자2): 문자2를 기준으로 문자1을 자른다.
text=input('날짜입력 yyyy.mm.dd') #input을 text라는 변수에 넣기
y,m,d=text.split('.') #'.'으로 바꾸고 각각을 y,m,d라는 변수에 넣기

print(text,y,m,d)
'''

'''
#map(함수, 집합 형태-iterable객체)
a,b,c=map(int,['1','2','3']) #list 형태 # 집합 안에 있는 각각을 다 int에 넣어줌
print(a,b,c,a+b+c)
'''


#하나씩 적용

a,b,c=map(int.input('a b c값 입력').split())

text=input('a b c값 입력')
text=text.split()
a,b,c=map(int,text)

print(a,b,c,a+b+c)

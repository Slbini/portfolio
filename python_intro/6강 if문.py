#6강 if문

#[목차]
#1. if문
#2. if~else
#3. if~elif
#4. if~elif~else



#1. if문
#특정 조건을 만족할 때와 만족하지 않을 때를 나누어 코드를 실행하는 조건문
#조건이 참일 때 = 실행
#조건이 거짓일 때 = 실행하지 않음
#if 조건: #한 칸 띄어쓰기
#   실행코드 #한 블럭 들여쓰기
'''
num=int(input('숫자 하나 입력:'))

if num>0 :
    print('{}은(는) 양수입니다.'.format(num))
'''



#2. if~else
'''
num=int(input('숫자 하나 입력:'))

if num%2==0 :
    print('{}은(는) 짝수입니다.'.format(num))
else :
    print('{}은(는) 홀수입니다.'.format(num))
'''




#3. if~elif
'''
age=int(input('나이 입력:'))

if age<=7 :
    print('유아입니다')
elif age<=19 :
    print('청소년입니다')
elif age>=20 :
    print('성인입니다')    
#위에서 부터 검사하게 되는데 위에서 true나오면 뒤에 검사 안 함
#모두 if문으로 넣으면 두 개 나옴
'''



#4. if~elif~else

text=input('알파벳 입력:')

if text.isupper() :
    print('대문자')
elif text.islower() :
    print('소문자')
else :
    print('대/소문자 구분 불가능')


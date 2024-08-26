#5 문자열 다루기
#[목차]
#1. 문자열 인덱스
#2. 문자열 슬라이스
#3. 문자열 메소드




#1. 문자열 인덱스
'''
text='abc' #0번칸, 1번칸, 2번칸

print(text[0]) #text를 불러오면 전체 다 불러오는 거고, 번호를 매겨서 불러오면 그 칸에 있는 하나의 문자만 불러 
print(text[1])
print(text[2])
#print(text[3])

print(text[-3]) 
print(text[-2])
print(text[-1]) #음수로 하면 -n부터 시작
#float 오면 안 됨
#print(text[-2.0])
'''


#2. 문자열 슬라이스
'''
text='abcde fgh ijk' #공백까지 생각해서 숫자 세어줘야 함
print(text[2:5])
print(text[1:8]) #자르고 싶은 번호 뒤 번호를 적어줘야 함
#8번까지 출력하는게 아니라 7번까지 출력함
print(text[-5:-1])
print(text[5:])
print(text[:5])
print(text[:]) #처음부터 끝까지
print(text[0:8:2]) #세번째:가져올 문자들의 간격
#print(text[8:0:-1]) #범위만큼 거꾸로 가져옴 #0번칸 뒤, 즉 1번칸까지만 출력 됨
print(text[::-1]) #처음부터 끝까지 뒤에서부터 출력
'''




#3. 문자열 메소드
#1.출력지정 : format(a,b,c,...)
'''
text = 'abcde {} {}'
print(text.format('ABC',123))
print(text.format('ABC',123, '???')) #넘어가는 건 잘라서 출력
#print(text.format('ABC')) #모자라는건 에러
'''

#2. 대체하기 : replace(a,b)
'''
text = 'abcde ABC ABC'
print(text.replace('A','K'))
'''

#3. 자르기: split(a)
'''
text = 'abcde A/B/C A.B.C'
a,b,c=text.split('.')
print(a)
print(b)
print(c)
'''

#4. 합치기 : a.join()
'''
text = 'abcde'
print('/'.join(text))
'''

#5. 개수 확인하기 : count(a)
'''
text = 'abcde ABC ABC'
print(text.count('a'))
print(text.count('A'))
print(text.count('1'))
'''

#6. 제거하기 : strip(a) / lstrip(a) / rstrip(a)
'''
text = '**abcde**'
print(text.strip('*'))
print(text.lstrip('*'))
print(text.rstrip('*'))
'''

#7. 인덱스 찾기 : find(a) / rfind(a) / index(a)/ rindex(a)
'''
text = 'ABC ABC'
print(text.find('A'))
print(text.rfind('A'))
print(text.index('A'))
print(text.rindex('A'))
text = 'ABC ABC'
print(text.find('d'))
print(text.rfind('d')) #없는거 찾으라하면 find는 -1출력 # index는 에러
print(text.index('d'))
print(text.rindex('d'))
'''

#8. 확인하기 : isalpha()/ isdigit()/ isalnum():숫자와 알파벳으로만?
# isupper():  대문자로 이루어져 있는지 확/ islower(): 소문자
#True or False 로만 대답
'''
text1 = 'ABCabc123'
text2 = '123'
text3 = 'ABC'
text4 = 'abc'
print(text3.isalpha())
print(text3.isdigit())
print(text3.isalnum()) #ture
print(text3.isupper())
print(text3.islower())
'''

#9. 대/소문자 만들기 : upper()/ lower()
'''
text = 'ABCabc'
print(text.upper())
print(text.lower())
'''

#10. 0 채우기 : zfill()
#자리수를 넣어줌
y='2020'
m='3'
d='1'
print(y.zfill(4))
print(m.zfill(2))
print(d.zfill(2))

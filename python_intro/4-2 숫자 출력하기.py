#숫자 출력하기
x=3
y=5
#print(x,y,x+y)
print('3과 5의 합은 8이다.')




# 숫자와 문자 함께 출력하기 (1) 콤마 & 형변환

#print(x,'과',y,'의 합은',x+y,'이다.')
#print(str(x)+'과 '+str(y)+'의 합은 '+str(x+y)+'이다.')


'''
# 숫자와 문자 함께 출력하기 (2) end=''
# 줄바꿈을 없애는 것

print(x,end='')
print('과 ',end='')
print(y,end='')
print('의 합은 ',end='')
print(x+y,end='')
print('이다')
'''



# 숫자와 문자 함께 출력하기 (3) format()

print('{}과 {}의 합은 {}이다.'.format(x,y,x+y))

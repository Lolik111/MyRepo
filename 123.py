n = 20
N = 1000

C = np.random.random_sample((n, n))
C = - C.dot(C.T) + 1000 * np.eye(n)
# print(np.linalg.eigvals(C))

import cvxpy as cp

mX = cp.Symmetric(n, n)
constraints = [cp.lambda_min(mX) >= 0, cp.diag(mX) == 1]

obj = cp.Maximize(cp.trace(C * mX))
maxcut = cp.Problem(obj, constraints).solve()

print(maxcut)
# print(mX.value)

data = np.zeros(N)
for i in range(N):
    X = np.matrix(np.random.randint(0, 2, n) * 2 - 1).T
    data[i] = X.T.dot(C).dot(X)[0][0]

fig = plt.figure(figsize=[10, 5])

plt.plot(data, 'bo', markersize=1)
plt.plot([maxcut], 'ro')
plt.plot([maxcut for i in range(N)], 'ro', markersize=0.2)


plt.show()
 V = np.matrix(np.linalg.cholesky(mX.value))

dataGW = np.zeros(N)

for i in range(N):
    r = np.random.random_sample(n)
    r = np.matrix(r / np.linalg.norm(r)).T
    
    x = np.matrix(np.sign(V.T.dot(r)))
    dataGW[i] = x.T.dot(C).dot(x)[0][0]

maxcutGW = np.array(dataGW).mean()
print(maxcut)
print(maxcutGW)

fig = plt.figure(figsize=[10, 5])

plt.plot(data, 'bo', markersize=1)
plt.plot(maxcut, 'ro')
plt.plot([maxcut for i in range(N)], 'ro', markersize=0.2)
plt.plot(dataGW, 'ko', markersize=1)
plt.plot(maxcutGW, 'mo')
plt.plot([maxcutGW for i in range(N)], 'mo', markersize=0.2)
# plt.plot(np.array(data).mean(), 'yo')
# plt.plot([np.array(data).mean() for i in range(N)], 'yo', markersize=0.2)

plt.show()

## Convex Optimization Practical Problem
## Name - Preet Paul
## Registration Number - 23414110019

## Question 1
## f(x1,x2) = (x1 - 2)^2 + (x2 - 5)^2
## Write a code in R to -
##(i) Use gradient descent to minimize f(x1, x2).
##(ii) Plot the solution path in a 2D contour plot.

t = 0.1

f = function(x)
{
	((x[1] - 2)^2) + ((x[2] - 5)^2)
}

g = function(x)
{
	c(2*(x[1] - 2), 2*(x[2] - 5))
}

x = matrix(NA, nrow = 2, ncol = 1001)
x[,1] = c(0,0)

for (i in 1:1000)
{
	x[,i+1] = x[,i] - (t * g(x[,i]))
}

x_val = seq(-1, 10, length.out = 500)
y_val = seq(-1,10, length.out = 500)
z = outer(x_val,y_val, Vectorize(function(x1, x2) f(c(x1, x2))))

contour(x_val, y_val, z, nlevels = 50)
points(x[1,], x[2,])


library(devtools)

x = rnorm(500, 5, 1)
y = 3 + (1*x)

g = function(theta)
{
	a = theta[1]
	b = theta[2]
	e = y - (a + b * x)
	g1 = -2 * sum(e)
	g2 = -2 * sum(x * e)
	c(g1, g2)
}

f = function(x, y, alpha, beta)
{
	return(sum((y - alpha - beta * x)^2))
}

t = 0.000001

m = matrix(NA, nrow = 2, ncol = 100001)
m[,1] = c(0,0)
for (i in 1:100000)
{
	m[,i+1] = m[,i] - (t * g(m[,i]))
}

## Plotting the gradient descent

plot(m[1,],m[2,])

## Plotting the contour plot

alpha_val = seq(-5, 10, length.out = 500)
beta_val = seq(-5, 10, length.out = 500)
z = outer(alpha_val, beta_val, Vectorize(function(a,b) f(x, y, a, b)))
contour(alpha_val, beta_val, z, nlevels = 50)
points(m[1,], m[2,])














































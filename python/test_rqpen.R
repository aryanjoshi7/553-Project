library(devtools)
devtools::load_all("/Users/micahwilliamson/code/EECS553/553-Project/rqpen")

n <- 20
p <- 3
x0<- matrix(rnorm(n*p),n,p)
x<- cbind(x0, x0^2, x0^3)[,order(rep(1:p,3))]
y<- -2+x[,1]+0.5*x[,2]-x[,3]-0.5*x[,7]+x[,8]-0.2*x[,9]+rt(n,2)
group<- rep(1:p, each=3)
# rq.print_test()
# lasso estimation
# one tau
# fit1 <- rq.pen(x,y)
# # several values of tau
# fit2 <- rq.pen(x,y,tau=c(.2,.5,.8))

# # # Group SCAD estimation
#fit3 <- rq.group.pen(x,y,groups=group,penalty="gSCAD")
# print(coefficients(fit3)[-1])
# # cross validation
# cv1 <- rq.pen.cv(x,y)
# plot(cv1)

# cv2 <- rq.pen.cv(x,y,tau=c(.2,.5,.8))
# plot(cv2)

# cv3 <- rq.group.pen(x,y,groups=group,penalty="gSCAD")
# plot(cv3)

# # BIC selection of tuning parameters
# qs1 <- qic.select(fit1)
# qs2 <- qic.select(fit2)
# qs3 <- qic.select(fit3)
function yp=myFlow(t,y)

ys=max(-1, min(y(1)^2,1)); %saturated y(1)^2
yp1=ys+y(2);
yp2=0;

yp=[yp1; yp2];

end
%判断是否为情况(2)的函数，若为情况(2)则返回1，其它返回0。输入参数为(n,temp)
function res=if2(n,temp)%(x,y,z,l,w,theta)
    if n==1
        res=0;
        return
    else
        for ii=1:n
            for jj=(ii+1):n
                x1=temp(6*ii-5);
                y1=temp(6*ii-4);
                z1=temp(6*ii-3);
                l1=temp(6*ii-2);
                w1=temp(6*ii-1);
                theta1=temp(6*ii);
                x2=temp(6*jj-5);
                y2=temp(6*jj-4);
                z2=temp(6*jj-3);
                l2=temp(6*jj-2);
                w2=temp(6*jj-1);
                theta2=temp(6*jj);
                if ((z1+w1/2)<(z2-w2/2))||((z1-w1/2)<(z2+w2/2))
                    continue
                else
                    if ifif2(x1,y1,l1,w1,theta1,x2,y2,l2,w2,theta2)
                        res=1;
                        return
                    end
                end
            end
        end
    end
    res=0;
    return
end

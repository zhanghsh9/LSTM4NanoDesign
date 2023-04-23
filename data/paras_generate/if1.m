%判断是否为情况(1)的函数，若为情况(1)则返回1，其它返回0。输入参数为px,py,rangeZ,n,temp)
function res=if1(px,py,rangeZ,n,temp)%(px,py,rangeZ,x,y,z,l,w,theta)
    for ii=1:n
        x=temp(6*ii-5);
        y=temp(6*ii-4);
        z=temp(6*ii-3);
        l=temp(6*ii-2);
        w=temp(6*ii-1);
        theta=temp(6*ii);
        if ifif1(px,py,rangeZ,x,y,z,l,w,theta)
            res=1;
            return
        end
    end
    res=0;
    return
end
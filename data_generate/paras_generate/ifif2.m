%判断一组参数是否为情况(2)的函数，若为情况(2)则返回1，其它返回0。输入参数为(px,
%py,x1,y1,z1,l1,w1,theta1,x1,y1,z1,l1,w1,theta1)
function res=ifif2(x1,y1,l1,w1,theta1,x2,y2,l2,w2,theta2)
    a1x = x1+(l1/2)*cosd(theta1)-(w1/2)*sind(theta1);
    a1y = y1+(l1/2)*sind(theta1)+(w1/2)*cosd(theta1);
    b1x = x1+(l1/2)*cosd(theta1)+(w1/2)*sind(theta1);
    b1y = y1+(l1/2)*sind(theta1)-(w1/2)*cosd(theta1);
    c1x = x1-(l1/2)*cosd(theta1)+(w1/2)*sind(theta1);
    c1y = y1-(l1/2)*sind(theta1)-(w1/2)*cosd(theta1);
    d1x = x1-(l1/2)*cosd(theta1)-(w1/2)*sind(theta1);
    d1y = y1-(l1/2)*sind(theta1)+(w1/2)*cosd(theta1);
    a2x = x2+(l2/2)*cosd(theta2)-(w2/2)*sind(theta2);
    a2y = y2+(l2/2)*sind(theta2)+(w2/2)*cosd(theta2);
    b2x = x2+(l2/2)*cosd(theta2)+(w2/2)*sind(theta2);
    b2y = y2+(l2/2)*sind(theta2)-(w2/2)*cosd(theta2);
    c2x = x2-(l2/2)*cosd(theta2)+(w2/2)*sind(theta2);
    c2y = y2-(l2/2)*sind(theta2)-(w2/2)*cosd(theta2);
    d2x = x2-(l2/2)*cosd(theta2)-(w2/2)*sind(theta2);
    d2y = y2-(l2/2)*sind(theta2)+(w2/2)*cosd(theta2);

    ab1mx=(a1x+b1x)/2;
    ab1my=(a1y+b1y)/2;
    ad1mx=(a1x+d1x)/2;
    ad1my=(a1y+d1y)/2;
    bc1mx=(b1x+c1x)/2;
    bc1my=(b1y+c1y)/2;
    cd1mx=(c1x+d1x)/2;
    cd1my=(c1y+d1y)/2;
    ab2mx=(a2x+b2x)/2;
    ab2my=(a2y+b2y)/2;
    ad2mx=(a2x+d2x)/2;
    ad2my=(a2y+d2y)/2;
    bc2mx=(b2x+c2x)/2;
    bc2my=(b2y+c2y)/2;
    cd2mx=(c2x+d2x)/2;
    cd2my=(c2y+d2y)/2;

    %求第一个矩形每条边的直线方程
    k1=tand(theta1);
    t11=a1y-(k1*a1x);
    t12=b1y+(b1x/k1);
    t13=c1y-(k1*c1x);
    t14=d1y+(d1x/k1);

    %求第一个矩形中线方程
    t15=(t11+t13)/2;
    t16=(t12+t14)/2;

    %求第二个矩形每条边的直线方程
    k2=tand(theta2);
    t21=a2y-(k2*a2x);
    t22=b2y+(b2x/k2);
    t23=c2y-(k2*c2x);
    t24=d2y+(d2x/k2);

    %求第二个矩形中线方程
    t25=(t21+t23)/2;
    t26=(t22+t24)/2;
    
    %判断第一个矩形内有没有第二个矩形的点
    if ((((k1*a2x)-a2y+t11)*((k1*x1)-y1+t11))>0)
        if (((((-1/k1)*a2x)-a2y+t12)*(((-1/k1)*x1)-y1+t12))>0)
            if ((((k1*a2x)-a2y+t13)*((k1*x1)-y1+t13))>0)
                if (((((-1/k1)*a2x)-a2y+t14)*(((-1/k1)*x1)-y1+t14))>0)
                    res=1;
                    return
                end
            end
        end
    end

    if ((((k1*b2x)-b2y+t11)*((k1*x1)-y1+t11))>0)
        if (((((-1/k1)*b2x)-b2y+t12)*(((-1/k1)*x1)-y1+t12))>0)
            if ((((k1*b2x)-b2y+t13)*((k1*x1)-y1+t13))>0)
                if (((((-1/k1)*b2x)-b2y+t14)*(((-1/k1)*x1)-y1+t14))>0)
                    res=1;
                    return
                end
            end
        end
    end

    if ((((k1*c2x)-c2y+t11)*((k1*x1)-y1+t11))>0)
        if (((((-1/k1)*c2x)-c2y+t12)*(((-1/k1)*x1)-y1+t12))>0)
            if ((((k1*c2x)-c2y+t13)*((k1*x1)-y1+t13))>0)
                if (((((-1/k1)*c2x)-c2y+t14)*(((-1/k1)*x1)-y1+t14))>0)
                    res=1;
                    return
                end
            end
        end
    end

    if ((((k1*d2x)-d2y+t11)*((k1*x1)-y1+t11))>0)
        if (((((-1/k1)*d2x)-d2y+t12)*(((-1/k1)*x1)-y1+t12))>0)
            if ((((k1*d2x)-d2y+t13)*((k1*x1)-y1+t13))>0)
                if (((((-1/k1)*d2x)-d2y+t14)*(((-1/k1)*x1)-y1+t14))>0)
                    res=1;
                    return
                end
            end
        end
    end

    %判断第二个矩形内有没有第一个矩形的点
    if ((((k2*a1x)-a1y+t21)*((k2*x2)-y2+t21))>0)
        if (((((-1/k2)*a1x)-a1y+t22)*(((-1/k2)*x2)-y2+t22))>0)
            if ((((k2*a1x)-a1y+t23)*((k2*x2)-y2+t23))>0)
                if (((((-1/k2)*a1x)-a1y+t24)*(((-1/k2)*x2)-y2+t24))>0)
                    res=1;
                    return
                end
            end
        end
    end

    if ((((k2*b1x)-b1y+t21)*((k2*x2)-y2+t21))>0)
        if (((((-1/k2)*b1x)-b1y+t22)*(((-1/k2)*x2)-y2+t22))>0)
            if ((((k2*b1x)-b1y+t23)*((k2*x2)-y2+t23))>0)
                if (((((-1/k2)*b1x)-b1y+t24)*(((-1/k2)*x2)-y2+t24))>0)
                    res=1;
                    return
                end
            end
        end
    end

    if ((((k2*c1x)-c1y+t21)*((k2*x2)-y2+t21))>0)
        if (((((-1/k2)*c1x)-c1y+t22)*(((-1/k2)*x2)-y2+t22))>0)
            if ((((k2*c1x)-c1y+t23)*((k2*x2)-y2+t23))>0)
                if (((((-1/k2)*c1x)-c1y+t24)*(((-1/k2)*x2)-y2+t24))>0)
                    res=1;
                    return
                end
            end
        end
    end

    if ((((k2*d1x)-d1y+t21)*((k2*x2)-y2+t21))>0)
        if (((((-1/k2)*d1x)-d1y+t22)*(((-1/k2)*x2)-y2+t22))>0)
            if ((((k2*d1x)-d1y+t23)*((k2*x2)-y2+t23))>0)
                if (((((-1/k2)*d1x)-d1y+t24)*(((-1/k2)*x2)-y2+t24))>0)
                    res=1;
                    return
                end
            end
        end
    end
    if k1~=k2
        tempx=(t25-t15)./(k1-k2);
        tempy=k1*tempx+t15;
        if ((((k2*tempx)-tempy+t21)*((k2*x2)-y2+t21))>0)
            if (((((-1/k2)*tempx)-tempy+t22)*(((-1/k2)*x2)-y2+t22))>0)
                if ((((k2*tempx)-tempy+t23)*((k2*x2)-y2+t23))>0)
                    if (((((-1/k2)*tempx)-tempy+t24)*(((-1/k2)*x2)-y2+t24))>0)
                        if ((((k1*tempx)-tempy+t11)*((k1*x1)-y1+t11))>0)
                            if (((((-1/k1)*tempx)-tempy+t12)*(((-1/k1)*x1)-y1+t12))>0)
                                if ((((k1*tempx)-tempy+t13)*((k1*x1)-y1+t13))>0)
                                    if (((((-1/k1)*tempx)-tempy+t14)*(((-1/k1)*x1)-y1+t14))>0)
                                        res=1;
                                        return
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    if k1~=k2
        tempx=(t26-t16)./(1/k2-1/k1);
        tempy=-1/k1*tempx+t16;
        if ((((k2*tempx)-tempy+t21)*((k2*x2)-y2+t21))>0)
            if (((((-1/k2)*tempx)-tempy+t22)*(((-1/k2)*x2)-y2+t22))>0)
                if ((((k2*tempx)-tempy+t23)*((k2*x2)-y2+t23))>0)
                    if (((((-1/k2)*tempx)-tempy+t24)*(((-1/k2)*x2)-y2+t24))>0)
                        if ((((k1*tempx)-tempy+t11)*((k1*x1)-y1+t11))>0)
                            if (((((-1/k1)*tempx)-tempy+t12)*(((-1/k1)*x1)-y1+t12))>0)
                                if ((((k1*tempx)-tempy+t13)*((k1*x1)-y1+t13))>0)
                                    if (((((-1/k1)*tempx)-tempy+t14)*(((-1/k1)*x1)-y1+t14))>0)
                                        res=1;
                                        return
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    res=0;
    return
end
%判断一组参数是否为情况(1)的函数，若为情况(1)则返回1，其它返回0。输入参数为(px,py,x,y,z,l,w,theta)
function res=ifif1(px,py,rangeZ,x,y,z,l,w,theta)
    if (z+w/2>max(rangeZ))||(z-w/2<min(rangeZ))
        res = 1;
        return
    else
        ax = x+(l/2)*cosd(theta)-(w/2)*sind(theta);
        ay = y+(l/2)*sind(theta)+(w/2)*cosd(theta);
        bx = x+(l/2)*cosd(theta)+(w/2)*sind(theta);
        by = y+(l/2)*sind(theta)-(w/2)*cosd(theta);
        cx = x-(l/2)*cosd(theta)+(w/2)*sind(theta);
        cy = y-(l/2)*sind(theta)-(w/2)*cosd(theta);
        dx = x-(l/2)*cosd(theta)-(w/2)*sind(theta);
        dy = y-(l/2)*sind(theta)+(w/2)*cosd(theta);
        temp1=[ax bx cx dx];
        temp2=[ay by cy dy];
        p1 = (px/2)*ones(1,4);
        p2 = (py/2)*ones(1,4);
        cond1 = sum(temp1>p1);
        cond2 = sum(temp1<(-1*p1));
        cond3 = sum(temp2>p2);
        cond4 = sum(temp2<(-1*p2));
        if (cond1+cond2+cond3+cond4)==0
            res = 0;
            return
        else
            res = 1;
            return
        end
    end
end
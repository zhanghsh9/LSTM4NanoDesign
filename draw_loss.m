clear all;
final_loss_forward=[];
final_loss_backward=[];
attn=0:0.5:5.5;
for ii=attn
    load(['results\20230426\loss_attn' num2str(ii) '.mat'])
    final_loss_forward=[final_loss_forward, vloss_record(end)];
    load(['results\20230426\loss_attn_backward' num2str(ii) '.mat'])
    final_loss_backward=[final_loss_backward, vloss_record(end)];
end

figure()
plot(attn,final_loss_forward,'LineWidth',2)
xlabel('Attention','FontSize',15)
ylabel('MSE','FontSize',15)
legend('forward')
title('MSE to attention, forward','FontSize',15)

figure()
plot(attn,final_loss_backward,'LineWidth',2)
xlabel('Attention','FontSize',15)
ylabel('MSE','FontSize',15)
legend('backward')
title('MSE to attention, backward','FontSize',15)


forward_enhance_1=(min(final_loss_forward)-final_loss_forward(3))/final_loss_forward(3);
forward_enhance_0=(min(final_loss_forward)-final_loss_forward(1))/final_loss_forward(1);

backward_enhance_1=(min(final_loss_backward)-final_loss_backward(3))/final_loss_backward(3);
backward_enhance_0=(min(final_loss_backward)-final_loss_backward(1))/final_loss_backward(1);

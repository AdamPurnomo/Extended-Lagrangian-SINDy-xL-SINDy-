function dydt=SpherePenODE(t,y,m,l,g)


theta = y(1,:);
psi = y(2,:);
theta_t = y(3,:);
psi_t = y(4,:);
theta_2t = sin(theta).*cos(theta).*psi_t.^2 - (g./l).*sin(theta);
psi_2t = -2.*theta_t.*psi_t.*cos(theta)./sin(theta);
dydt=[
    theta_t;
    psi_t;
    theta_2t;
    psi_2t];

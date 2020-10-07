function [w,l,lcomp] = FindInitGuess(w,A,b)
    relu = @(w)max(w,0);
    drelu = @(w)ones(size(w)).*sign(relu(w));
    l = sum(relu(b - A*w));
    iter = 0;
    itermax = 10000;
    step = 0.1;
    while l > 0 && iter < itermax
      dl = sum(-drelu(b - A*w)'*A,1)';
      if norm(dl) > 1
          dl = dl/norm(dl);
      end
      w = w - step*dl;
      l = sum(relu(b - A*w));
      iter = iter + 1;
    end
fprintf('%d iterations: x = [%d,%d,%d,%d],f = %d\n',iter, w(1),w(2),w(3),w(4),l);
lcomp = relu(b - A*w);
end
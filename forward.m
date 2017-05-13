function predictions = predict(W1,W2, X)
  W{1} = W1;
  W{2} = W2;
  outputs{1} = X;
  for i = 1:length(W)
    if (i == length(W))
      outputs{i + 1} = forward_layer(W{i}, outputs{i}, @id);
    else
      outputs{i + 1} = forward_layer(W{i}, outputs{i}, @sigmoid);
    endif
  endfor
  predictions = outputs{length(outputs)};
endfunction

function x = id(x)
endfunction

function y = sigmoid(z)
  y = 1 ./ (1 + e.^-z);
endfunction

function Y = forward_layer(W, X, f)
  Y = f (X * W);
endfunction
 
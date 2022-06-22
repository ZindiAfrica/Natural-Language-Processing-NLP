#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

cd $HOME/solution/models

# 10 GB
cd run-20210508-1501-tf-mt5large-len144-b24-7e5-4v100
echo "Downloading weights. Model 1 of 6..."
curl -L -o model-best-f0-e016-0.6458.h5 https://www.dropbox.com/s/7djl031walx5uqq/model-best-f0-e016-0.6458.h5?dl=0
curl -L -o model-best-f1-e011-0.6376.h5 https://www.dropbox.com/s/t0kq3q41y0vpea3/model-best-f1-e011-0.6376.h5?dl=0
curl -L -o model-best-f2-e013-0.6829.h5 https://www.dropbox.com/s/c7zqjybgj3nmf6h/model-best-f2-e013-0.6829.h5?dl=0
curl -L -o model-best-f3-e012-0.7213.h5 https://www.dropbox.com/s/1pvl9vydybu7jfh/model-best-f3-e012-0.7213.h5?dl=0
curl -L -o model-best-f4-e019-0.6794.h5 https://www.dropbox.com/s/sqhtx805mp9noum/model-best-f4-e019-0.6794.h5?dl=0
cd ..

# 10 GB
cd run-20210508-1646-tf-mt5large-len256-b24-7e5-4v100
echo "Downloading weights. Model 2 of 6..."
curl -L -o model-best-f0-e019-0.6424.h5 https://www.dropbox.com/s/y0uufbjuezfevdv/model-best-f0-e019-0.6424.h5?dl=0
curl -L -o model-best-f1-e013-0.6690.h5 https://www.dropbox.com/s/42qi0rn4ivyg566/model-best-f1-e013-0.6690.h5?dl=0
curl -L -o model-best-f2-e014-0.6969.h5 https://www.dropbox.com/s/hkse56crwpuhvov/model-best-f2-e014-0.6969.h5?dl=0
curl -L -o model-best-f3-e019-0.6620.h5 https://www.dropbox.com/s/uehfu8t1aj53h19/model-best-f3-e019-0.6620.h5?dl=0
curl -L -o model-best-f4-e015-0.6620.h5 https://www.dropbox.com/s/q6e24skbd4knrhy/model-best-f4-e015-0.6620.h5?dl=0
cd ..

# 10 GB
cd run-20210509-1534-tf-mt5large-len64-b24-7e5-4v100
echo "Downloading weights. Model 3 of 6..."
curl -L -o model-best-f0-e020-0.6458.h5 https://www.dropbox.com/s/xeini59p30nf869/model-best-f0-e020-0.6458.h5?dl=0
curl -L -o model-best-f1-e018-0.6132.h5 https://www.dropbox.com/s/6z8qk47gsv5an33/model-best-f1-e018-0.6132.h5?dl=0
curl -L -o model-best-f2-e016-0.6516.h5 https://www.dropbox.com/s/pnsjohojt3wd4e6/model-best-f2-e016-0.6516.h5?dl=0
curl -L -o model-best-f3-e020-0.6864.h5 https://www.dropbox.com/s/13wki9bqsfc7bg4/model-best-f3-e020-0.6864.h5?dl=0
curl -L -o model-best-f4-e020-0.6516.h5 https://www.dropbox.com/s/tn2jfz4k6bdn8ez/model-best-f4-e020-0.6516.h5?dl=0
cd ..

# 10 GB
cd run-20210509-1605-tf-mt5large-len128-b24-7e5-4p40
echo "Downloading weights. Model 4 of 6..."
curl -L -o model-best-f0-e017-0.6667.h5 https://www.dropbox.com/s/ueb9njuvooy0y88/model-best-f0-e017-0.6667.h5?dl=0
curl -L -o model-best-f1-e020-0.6725.h5 https://www.dropbox.com/s/49d65l5w98ezuzh/model-best-f1-e020-0.6725.h5?dl=0
curl -L -o model-best-f2-e018-0.6620.h5 https://www.dropbox.com/s/ppdwdiadsl6du4n/model-best-f2-e018-0.6620.h5?dl=0
curl -L -o model-best-f3-e018-0.7247.h5 https://www.dropbox.com/s/vdv1ip7q2ntd3am/model-best-f3-e018-0.7247.h5?dl=0
curl -L -o model-best-f4-e010-0.6516.h5 https://www.dropbox.com/s/6d54mmkkaqdbtyr/model-best-f4-e010-0.6516.h5?dl=0
cd ..

# 10 GB
cd run-20210509-1614-tf-mt5large-len192-b24-7e5-4v100
echo "Downloading weights. Model 5 of 6..."
curl -L -o model-best-f0-e010-0.6458.h5 https://www.dropbox.com/s/z9h1u0r2079tyfv/model-best-f0-e010-0.6458.h5?dl=0
curl -L -o model-best-f1-e019-0.6655.h5 https://www.dropbox.com/s/t89hiqkr1n37h4b/model-best-f1-e019-0.6655.h5?dl=0
curl -L -o model-best-f2-e012-0.6655.h5 https://www.dropbox.com/s/v1fhp69v7p5wlw1/model-best-f2-e012-0.6655.h5?dl=0
curl -L -o model-best-f3-e015-0.7387.h5 https://www.dropbox.com/s/9p4rqjxy47j6v05/model-best-f3-e015-0.7387.h5?dl=0
curl -L -o model-best-f4-e011-0.6655.h5 https://www.dropbox.com/s/xczce9wr09csjsa/model-best-f4-e011-0.6655.h5?dl=0
cd ..

# 32 GB
cd run-20210509-1735-tf-mt5xl-len128-b16-7e5-1a100
echo "Downloading weights. Model 6 of 6..."
curl -L -o model-best-f0-e015-0.6424.h5 https://www.dropbox.com/s/5l1bai7xx36od8r/model-best-f0-e015-0.6424.h5?dl=0
curl -L -o model-best-f1-e010-0.6585.h5 https://www.dropbox.com/s/jdy6zib9k0ou02o/model-best-f1-e010-0.6585.h5?dl=0
curl -L -o model-best-f2-e005-0.6272.h5 https://www.dropbox.com/s/ahudb0s8quv71fh/model-best-f2-e005-0.6272.h5?dl=0
curl -L -o model-best-f3-e005-0.6725.h5 https://www.dropbox.com/s/2g9u7oxsstg4wk2/model-best-f3-e005-0.6725.h5?dl=0
curl -L -o model-best-f4-e006-0.6167.h5 https://www.dropbox.com/s/0cf49j1cr0qghy3/model-best-f4-e006-0.6167.h5?dl=0
cd ..
cd ..

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------



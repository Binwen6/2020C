import rpy2.robjects as robjects

# 检查R的版本信息
version_info = robjects.r('version')
print(version_info)

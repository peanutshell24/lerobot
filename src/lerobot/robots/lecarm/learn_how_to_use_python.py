from functools import wraps
def a_new_dacorator(func):
    @wraps(func)
    def wrapTheFunction(name):
        print("I am here to be a dacorator !")
        func(name)
    return wrapTheFunction
@a_new_dacorator
def a_normal_function(name):
    print(f"I am a normal function, and this is {name.title()}!")

a_normal_function("jack")
print(a_normal_function.__name__)

def logit(logfile='out.log'):
    def logging_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            log_string = func.__name__ + " was called"
            print(log_string)
            # 打开logfile，并写入内容
            with open(logfile, 'a') as opened_file:
                # 现在将日志打到指定的logfile
                opened_file.write(log_string + '\n')
            return func(*args, **kwargs)
        return wrapped_function
    return logging_decorator

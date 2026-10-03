"""Own a Windows worker and every child before allowing its first instruction.

Windows venv Python is a redirector. Killing only its PID does not reliably stop
the actual tensor process, so use a suspended launch in a kill-on-close job.
"""
import ctypes
from ctypes import wintypes
import os
import subprocess


class WindowsOwnedProcess:
    def __init__(self, args):
        if os.name != 'nt':
            raise OSError('Windows process jobs are unavailable on this platform')
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel, self.job, self.handle = kernel, None, None
        self.returncode = None

        class Startup(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('reserved', wintypes.LPWSTR),
                        ('desktop', wintypes.LPWSTR), ('title', wintypes.LPWSTR),
                        ('x', wintypes.DWORD), ('y', wintypes.DWORD),
                        ('xsize', wintypes.DWORD), ('ysize', wintypes.DWORD),
                        ('xchars', wintypes.DWORD), ('ychars', wintypes.DWORD),
                        ('fill', wintypes.DWORD), ('flags', wintypes.DWORD),
                        ('show', wintypes.WORD), ('reserved_size', wintypes.WORD),
                        ('reserved_bytes', ctypes.c_void_p), ('stdin', wintypes.HANDLE),
                        ('stdout', wintypes.HANDLE), ('stderr', wintypes.HANDLE)]

        class ProcessInfo(ctypes.Structure):
            _fields_ = [('process', wintypes.HANDLE), ('thread', wintypes.HANDLE),
                        ('pid', wintypes.DWORD), ('tid', wintypes.DWORD)]

        class BasicLimits(ctypes.Structure):
            _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                        ('flags', wintypes.DWORD), ('min_working', ctypes.c_size_t),
                        ('max_working', ctypes.c_size_t), ('active_limit', wintypes.DWORD),
                        ('affinity', ctypes.c_size_t), ('priority', wintypes.DWORD),
                        ('scheduling', wintypes.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [('basic', BasicLimits), ('io', IO), ('process_memory', ctypes.c_size_t),
                        ('job_memory', ctypes.c_size_t), ('peak_process', ctypes.c_size_t),
                        ('peak_job', ctypes.c_size_t)]

        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.CreateProcessW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p,
                                        ctypes.c_void_p, wintypes.BOOL, wintypes.DWORD,
                                        ctypes.c_void_p, wintypes.LPCWSTR,
                                        ctypes.POINTER(Startup), ctypes.POINTER(ProcessInfo)]
        kernel.CreateProcessW.restype = wintypes.BOOL
        for name, argspec, result in (
            ('AssignProcessToJobObject', [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            ('ResumeThread', [wintypes.HANDLE], wintypes.DWORD),
            ('CloseHandle', [wintypes.HANDLE], wintypes.BOOL),
            ('TerminateJobObject', [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            ('TerminateProcess', [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            ('GetExitCodeProcess', [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)], wintypes.BOOL),
            ('WaitForSingleObject', [wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD),
        ):
            function = getattr(kernel, name)
            function.argtypes, function.restype = argspec, result
        info = ProcessInfo()
        try:
            self.job = kernel.CreateJobObjectW(None, None)
            if not self.job:
                raise ctypes.WinError(ctypes.get_last_error())
            limits = ExtendedLimits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel.SetInformationJobObject(self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise ctypes.WinError(ctypes.get_last_error())
            startup = Startup()
            startup.cb = ctypes.sizeof(startup)
            command = ctypes.create_unicode_buffer(subprocess.list2cmdline(args))
            if not kernel.CreateProcessW(args[0], command, None, None, False,
                                         0x00000004 | 0x08000000, None, None,
                                         ctypes.byref(startup), ctypes.byref(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            self.handle, self.pid = info.process, info.pid
            if not kernel.AssignProcessToJobObject(self.job, self.handle):
                # Still suspended: no unowned child could have started.
                kernel.TerminateProcess(self.handle, 1)
                raise ctypes.WinError(ctypes.get_last_error())
            if kernel.ResumeThread(info.thread) == 0xffffffff:
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            self.close()
            raise
        finally:
            if info.thread:
                kernel.CloseHandle(info.thread)

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        state = self.kernel.WaitForSingleObject(self.handle, 0)
        if state == 0x102:
            return None
        if state != 0:
            raise ctypes.WinError(ctypes.get_last_error())
        code = wintypes.DWORD()
        if not self.kernel.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        self.returncode = code.value
        return self.returncode

    def wait(self, timeout):
        state = self.kernel.WaitForSingleObject(self.handle, max(0, int(timeout * 1000)))
        if state == 0x102:
            raise subprocess.TimeoutExpired('analysis worker', timeout)
        return self.poll()

    def terminate(self):
        if self.job and not self.kernel.TerminateJobObject(self.job, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    kill = terminate

    def close(self):
        if self.job:
            self.kernel.CloseHandle(self.job)
            self.job = None
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def launch(args):
    if os.name == 'nt':
        return WindowsOwnedProcess(args)
    return subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)


def stop(process):
    if os.name == 'nt':
        process.terminate()
    else:
        import signal
        os.killpg(process.pid, signal.SIGKILL)

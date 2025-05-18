import asyncio
import logging
import time
import unicodedata
import pandas as pd
import shutil, psutil
from wcwidth import wcswidth
from abc import ABC, abstractmethod

from .connection import Connection
from .events.managers import ConnectorEventManager, WebsocketEventManager
from .exceptions import NoLeagueClientDetected
from .utils import _return_ux_process

logger = logging.getLogger('lcu-driver')

def count_nonASCII(s: str): #统计一个字符串中占用命令行2个宽度单位的字符个数（Count the number of characters that take up 2 width unit in CMD）
    return sum([unicodedata.east_asian_width(character) in ("F", "W") for character in list(str(s))])

def format_df(df: pd.DataFrame, width_exceed_ask: bool = True, direct_print: bool = False, print_header: bool = True, print_index: bool = False, reserve_index = False, start_index = 0, header_align: str = "^", align: str = "^", align_replicate_rule: str = "all"): #按照每列最长字符串的命令行宽度加上2，再根据每个数据的中文字符数量决定最终格式化输出的字符串宽度（Get the width of the longest string of each column, add it by 2, and substract it by the number of each cell string's Chinese characters to get the final width for each cell to print using `format` function）
    old_index = df.index #用于存储旧索引。当`reserve_index`为真时，将输出旧索引（Stores the old indices. When `reserve_index` is True, the program outputs the old indices）
    df.index = range(start_index, len(df) + start_index) #新索引允许从`start_index`开始，默认从0开始（New indices allow starting from `start_index`, which is 0 by default）
    maxLens = {} #存储不同列的最大字符串宽度（Stores the max string lengths of different columns）
    maxWidth = shutil.get_terminal_size()[0] #获取当前终端的单行宽度（Get the line width of the current terminal）
    fields = df.columns.tolist()
    for field in fields: #计算每一列的最大字符串宽度（Calculate the max string length of each column）
        maxLens[field] = max(0 if len(df) == 0 else max(map(lambda x: wcswidth(str(x)), df[field])), wcswidth(str(field))) + 2
    index_len = max(len(str(start_index)), len(str(start_index + len(df) - 1))) #计算每一列的最大字符串宽度（Calculate the max string length of the index column）
    if sum(maxLens.values()) + 2 * (len(fields) - 1) > maxWidth or print_index and index_len + sum(maxLens.values()) + 2 * len(fields) > maxWidth: #字符串宽度和超出终端窗口宽度的情形（The case where the sum of the string lengths exceeds the terminal size）
        if width_exceed_ask:
            print("单行数据字符串输出宽度超过当前终端窗口宽度！是否继续？（输入任意键继续，否则直接打印该数据框。）\nThe output width of each record string exceeds the current width of the terminal window! Continue? (Input anything to continue, or null to directly print this dataframe.)")
            if input() == "":
                #print(df)
                result = str(df)
                return (result, maxLens)
        elif direct_print:
            print("单行数据字符串输出宽度超过当前终端窗口宽度！将直接打印该数据框！\nThe output width of each record string exceeds the current width of the terminal window! The program is going to directly print this dataframe!")
            result = str(df)
            return (result, maxLens)
        else:
            print("单行数据字符串输出宽度超过当前终端窗口宽度！将继续格式化输出！\nThe output width of each record string exceeds the current width of the terminal window! The program is going on formatted printing!")
    result = "" #结果字符串初始化（Initialize the result string）
    #确定各列的排列方向（Determine the alignments of all columns）
    if isinstance(header_align, str) and isinstance(align, str): #确保排列方向参数无误（Ensure the alignment parameters are valid）·
        if not all(map(lambda x: x in {"<", "^", ">"}, header_align)) or not all(map(lambda x: x in {"<", "^", ">"}, align)):
            print('排列方式字符串参数错误！排列方式必须是“<”“^”或者“>”中的一个。请修改排列方式字符串参数。\nParameter ERROR of the alignment string! The alignment value must be one of {"<", "^", ">"}. Please change the alignment string parameter.')
        if len(header_align) == 0: #指定为空字符串，即默认居中输出（Specifying it as a null string means output centered by default）
            header_alignments = ["^"] * df.shape[1]
        elif len(header_align) == 1:
            header_alignments = [header_align] * df.shape[1]
        else:
            header_alignments_tmp = list(header_align)
            if len(header_align) < df.shape[1]: #表头排列规则字符串长度小于数据框列数时，通过排列方式列表补充规则进行补充（When the length of `header_align` is less than the number of the dataframe's columns, supplement the rest of the rules according to `align_replicate_rule`）
                if align_replicate_rule == "last": #仅重复最后一列的排列方式（Only replicate the alignment of the last column）
                    header_alignments = header_alignments_tmp + [header_alignments_tmp[-1]] * len(df.shape[1] - len(header_align))
                else:
                    if align_replicate_rule != "all":
                        print("排列方式列表补充规则不合法！将默认采用全部填充。\nAlignment list supplement rule illegal! The whole alignment string will be replicated.")
                    header_alignments = header_alignments_tmp * (df.shape[1] // len(header_align)) + header_alignments_tmp[:df.shape[1] % len(header_align)] #所有排列方式循环补充（Supplement the alignments in a cycle of the whole `header_alignment` string）
            else: #表头排列规则字符串大于等于数据框列数时，取长度等于数据框列数的字符串开头切片（When the length of `header_align` is greater than or equal to the number of the dataframe's columns, get the slice at the beginning of `header_align` whose length equal to the number of the dataframe's columns）
                header_alignments = header_alignments_tmp[:df.shape[1]]
        if len(align) == 0: #指定为空字符串，即默认居中输出（Specifying it as a null string means output centered by default）
            alignments = ["^"] * df.shape[1]
        elif len(align) == 1:
            alignments = [align] * df.shape[1]
        else:
            alignments_tmp = list(align)
            if len(align) < df.shape[1]: #数据排列规则字符串长度小于数据框列数时，通过排列方式列表补充规则进行补充（When the length of `align` is less than the number of the dataframe's columns, supplement the rest of the rules according to `align_replicate_rule`）
                if align_replicate_rule == "last": #仅重复最后一列的排列方式（Only replicate the alignment of the last column）
                    alignments = alignments_tmp + [alignments_tmp[-1]] * len(df.shape[1] - len(align))
                else:
                    if align_replicate_rule != "all":
                        print("排列方式列表补充规则不合法！将默认采用全部填充。\nAlignment list supplement rule illegal! The whole alignment string will be replicated.")
                    alignments = alignments_tmp * (df.shape[1] // len(align)) + alignments_tmp[:df.shape[1] % len(align)]
            else: #数据排列规则字符串大于等于数据框列数时，取长度等于数据框列数的字符串开头切片（When the length of `align` is greater than or equal to the number of the dataframe's columns, get the slice at the beginning of `header_align` whose length equal to the number of the dataframe's columns）
                alignments = alignments_tmp[:df.shape[1]]
        if print_header: #打印表头（Prints the header）
            if print_index: #打印表头时，如果输出索引，由于表头没有索引，所以用空格代替（Spaces will be printed as the index part of the header）
                result += " " * (index_len + 2)
            for i in range(df.shape[1]):
                field = fields[i]
                tmp = "{0:{align}{w}}".format(field, align = header_alignments[i], w = maxLens[str(field)] - count_nonASCII(str(field)))
                result += tmp
                #print(tmp, end = "")
                if i != df.shape[1] - 1: #未到行尾时，用两个空格来分割该列和下一列（When the program doesn't reach the end of the line, separate this column and the next column by two spaces）
                    result += "  "
                    #print("  ", end = "")
            result += "\n"
            #print()
        index = start_index
        for i in range(df.shape[0]):
            if print_index:
                result += "{0:>{w}}".format(old_index[index - start_index] if reserve_index else index, w = index_len) + "  "
            for j in range(df.shape[1]):
                field = fields[j]
                cell = str(list(df[field])[i])
                tmp = "{0:{align}{w}}".format(cell, align = alignments[j], w = maxLens[field] - count_nonASCII(cell))
                result += tmp
                #print(tmp, end = "")
                if j != df.shape[1] - 1: #未到行尾时，用两个空格来分割该列和下一列（When the program doesn't reach the end of the line, separate this column and the next column by two spaces）
                    result += "  "
                    #print("  ", end = "")
            if i != df.shape[0] - 1:
                result += "\n"
            #print() #注意这里的缩进和上一行不同（Note that here the indentation is different from the above line）
            index += 1
    else:
        print("排列方式参数错误！请传入字符串。\nAlignment parameter ERROR! Please pass a string instead.")
    return (result, maxLens)

class BaseConnector(ConnectorEventManager, ABC):
    def __init__(self, loop=None):
        super().__init__()
        self.loop = loop or asyncio.get_event_loop()
        self.ws = WebsocketEventManager()

    @abstractmethod
    def register_connection(self, connection: Connection):
        """Creates a connection and saves a reference to it"""
        pass

    @abstractmethod
    def unregister_connection(self, lcu_pid):
        """Cancel the connection"""
        pass

    @property
    def should_run_ws(self) -> bool:
        return True


class Connector(BaseConnector):
    def __init__(self, *, loop=None):
        super().__init__(loop)
        self._repeat_flag = True
        self.connection = None

    def register_connection(self, connection):
        self.connection = connection

    def unregister_connection(self, _):
        self.connection = None

    @property
    def should_run_ws(self) -> bool:
        return len(self.ws.registered_uris) > 0

    def start(self) -> None:
        """Starts the connector. This method should be overridden if different behavior is required.

        :rtype: None
        """
        try:
            def wrapper():
                process_iter = _return_ux_process()
                if len(process_iter) > 1:
                    print("检测到您运行了多个客户端。请选择您需要操作的客户端进程：（默认选择最近的进程）\nDetected multiple clients running. Please select a client process: (The latest process by default)")
                    process_dict = {"No.": ["序号"], "pid": ["进程序号"], "filePath": ["进程文件路径"], "createTime": ["进程创建时间"], "status": ["状态"]}
                    #process_header_df = pd.DataFrame(process_dict)
                    for i in range(len(process_iter)):
                        process_dict["No."].append(i + 1)
                        process_dict["pid"].append(process_iter[i].pid)
                        try:
                            process_dict["filePath"].append(process_iter[i].cmdline()[0])
                        except psutil.AccessDenied: #有时进程处于“已挂起”状态时，会无法访问Process类的cmdline、cwd、environ等方法（Sometimes when a process is suspended, attributes of a Process object, like `cmdline`, `cwd`, `environ`, etc., can't be accessed）
                            process_dict["filePath"].append(process_iter[i].exe())
                        process_dict["createTime"].append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(process_iter[i].create_time())))
                        process_dict["status"].append(process_iter[i].status())
                    process_df = pd.DataFrame(process_dict)
                    print(format_df(process_df, width_exceed_ask = False, direct_print = True)[0])
                    running_process_df = process_df[process_df.status == "running"].sort_values(by = ["createTime"], ascending = False)
                    while True:
                        processIndex = input()
                        if processIndex == "":
                            if len(running_process_df) == 0:
                                print("无活动英雄联盟客户端进程。程序即将退出！\nNo active LeagueClientUx process! The program will exit now!")
                                time.sleep(2)
                                exit(1)
                            else:
                                print("已选择最近创建的英雄联盟客户端进程。\nSelected the recently created LeagueClientUx process.")
                                processIndex = running_process_df.iat[0, 0]
                                print(format_df(process_df.iloc[[0, processIndex]])[0], end = "\n\n")
                        try:
                            processIndex = int(processIndex)
                        except ValueError:
                            print("请输入不超过%d的正整数！\nPlease input an integer not greater than %d!" %(len(process_iter), len(process_iter)))
                        else:
                            if processIndex in range(1, len(process_iter) + 1):
                                process = process_iter[processIndex - 1]
                                break
                            else:
                                print("请输入不超过%d的正整数！\nPlease input an integer not greater than %d!" %(len(process_iter), len(process_iter)))
                elif len(process_iter) == 1: #如果没有后面两个部分，那么在经过100次寻找进程后，由于process_iter中已经包含了所有符合要求的进程，process将成为None，从而导致self.loop.run_until_complete(connection.init())出现self中无_auth_keys的报错（If the following parts don't exist, then after 100 times of searching for the demanding process, since `process_iter` has included all the corresponding processes, `process` will become `None`, which causes an AttributeError that 'Connection' object has no attribute '_auth_key'）
                    process = process_iter[0]
                else:
                    raise NoLeagueClientDetected("The program didn't detect a running League Client.")
                connection = Connection(self, process)
                self.register_connection(connection)
                self.loop.run_until_complete(connection.init())

                if self._repeat_flag and len(self.ws.registered_uris) > 0:
                    logger.debug('Repeat flag=True. Looking for new clients.')
                    wrapper()

            wrapper()
        except KeyboardInterrupt:
            logger.info('Event loop interrupted by keyboard')
        self.loop.close()

    async def stop(self) -> None:
        """Flag the connector to don't look for more clients once the connection finishes his job.

        :rtype: None
        """
        self._repeat_flag = False
        if self.connection is not None:
            await self.connection._close()


class MultipleClientConnector(BaseConnector):
    def __init__(self, *, loop=None):
        super().__init__(loop=loop)
        self.connections = []

    def register_connection(self, connection):
        self.connections.append(connection)

    def unregister_connection(self, lcu_pid):
        for index, connection in enumerate(self.connections):
            if connection.pid == lcu_pid:
                del connection[index]

    @property
    def should_run_ws(self) -> bool:
        return True

    def _process_was_initialized(self, non_initialized_connection):
        for connection in self.connections:
            if non_initialized_connection.pid == connection.pid:
                return True
        return False

    async def _astart(self):
        tasks = []
        try:
            while True:
                process_iter = _return_ux_process()

                process = next(process_iter, None)
                while process:
                    connection = Connection(self, process)
                    if not self._process_was_initialized(connection):
                        tasks.append(asyncio.create_task(connection.init()))

                    process = next(process_iter, None)
                await asyncio.sleep(0.5)

        except KeyboardInterrupt:
            logger.info('Event loop interrupted by keyboard')
        finally:
            await asyncio.gather(*tasks)

    def start(self) -> None:
        self.loop.run_until_complete(self._astart())

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def create_market_analyst(llm, mcp_tools):
    """创建中国市场分析师"""

    def china_market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        # 从 mcp_tools 列表中查找需要的工具
        print(f"📊 [市场分析师] mcp_tools 类型: {type(mcp_tools)}, 长度: {len(mcp_tools) if mcp_tools else 0}")
        if mcp_tools:
            print(f"📊 [市场分析师] 可用工具: {[tool.name for tool in mcp_tools]}")

        tools = []
        for tool in mcp_tools:
            if tool.name == 'get_stock_market_data':
                tools.append(tool)
                print(f"✅ [市场分析师] 找到工具: {tool.name}")

        print(f"📊 [市场分析师] 最终 tools 列表长度: {len(tools)}")
        
        system_message = (
            """您是一位专业的中国股市分析师，专门分析A股、港股等中国资本市场。您具备深厚的中国股市知识和丰富的本土投资经验。

您的专业领域包括：
1. **A股市场分析**: 深度理解A股的独特性，包括涨跌停制度、T+1交易、融资融券等
2. **中国经济政策**: 熟悉货币政策、财政政策对股市的影响机制
3. **行业板块轮动**: 掌握中国特色的板块轮动规律和热点切换
4. **监管环境**: 了解证监会政策、退市制度、注册制等监管变化
5. **市场情绪**: 理解中国投资者的行为特征和情绪波动

分析重点：
- **技术面分析**: 使用通达信数据进行精确的技术指标分析
- **基本面分析**: 结合中国会计准则和财报特点进行分析
- **政策面分析**: 评估政策变化对个股和板块的影响
- **资金面分析**: 分析北向资金、融资融券、大宗交易等资金流向
- **市场风格**: 判断当前是成长风格还是价值风格占优

中国股市特色考虑：
- 涨跌停板限制对交易策略的影响
- ST股票的特殊风险和机会
- 科创板、创业板的差异化分析
- 国企改革、混改等主题投资机会
- 中美关系、地缘政治对中概股的影响

请基于Tushare数据接口提供的实时数据和技术指标，结合中国股市的特殊性，撰写专业的中文分析报告。
确保在报告末尾附上Markdown表格总结关键发现和投资建议。"""
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "您是一位专业的AI助手，与其他分析师协作进行股票分析。"
                    " 使用提供的工具获取和分析数据。"
                    " 如果您无法完全回答，没关系；其他分析师会补充您的分析。"
                    " 专注于您的专业领域，提供高质量的分析见解。"
                    " 您可以访问以下工具：{tool_names}。\n{system_message}"
                    "当前分析日期：{current_date}，分析标的：{ticker}。",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)
        
        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state['messages'])
        
        report = ""
        
        if len(result.tool_calls) == 0:
            report = result.content
        
        return {
            "messages": [result],
            "market_report": report,
        }
    
    return china_market_analyst_node

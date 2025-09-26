class loggingTemp:
    def __init__(self):
        self.reading = []
        self.prefixSum = [0]

    def addReading(self, temp: int):
        self.reading.append(temp)
        self.prefixSum.append(self.prefixSum[-1] + temp)    

    def getAverage(self, k: int) -> float:
        len_reading = len(self.reading)
        return (self.prefixSum[len_reading] - self.prefixSum[len_reading - k]) / k
        
    def getMaxWindow(self, k: int):
        n = len(self.reading)
        if k > n:
            raise ValueError("Not enough readings")
        for i in range(n - k + 1):
            window_sum = self.prefix[i + k] - self.prefix[i]
            if window_sum > max_sum:
                max_sum = window_sum
        return max_sum / k

